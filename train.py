'''
Author: Jieshan Chen
'''

import argparse, os, time, pickle, random, sys
import numpy as np
from tqdm import tqdm 

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from tensorboardX import SummaryWriter

from models.image_models import ResNetFeats
from opts import get_opt
from optim import NoamOpt
import models
from CenterLoss import CenterLoss
from dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from test import test
os.environ['CUDA_VISIBLE_DEVICES']='0'
torch.backends.cudnn.enabled = False

MEAN_TWO = (0.523307, 0.522698, 0.521938)
STD_TWO = (0.108646, 0.107423, 0.110502)

def main(args):
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Create result model directory
	args.model_path += "_" + args.caption_model
	if args.finetune_cnn:
		args.model_path += "_finetune"

	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	writer = SummaryWriter(log_dir=args.model_path)

	# Build data loader
	# dir_name = 'E:\\OSLAB\\LearnDependency\\data_bank\\'
	dir_name = '/mnt/data/zhangsk/LearnData/'
	file_name = 'train_one.txt'
	transform = transforms.Compose([transforms.ColorJitter(contrast=1),transforms.ColorJitter(hue=0.5),transforms.ColorJitter(brightness=1),transforms.ToTensor(), transforms.Normalize(MEAN_TWO, STD_TWO), transforms.ToPILImage()])
	train_data = MyDataset(root=dir_name, datatxt=file_name, transform=transform,target_transform=transforms.ToTensor())
	# weighted samples
	num_classes = 3
	class_sample_counts = [4642,6802,8668]
	# samples weight for possibility
	class_weights = 1./torch.Tensor(class_sample_counts)
	train_targets = [sample[4] for sample in train_data.imgs]
	train_samples_weight = [class_weights[class_id] for class_id in train_targets]
	# now lets initialize samplers
	train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(train_data))
	train_loader = DataLoader(dataset=train_data, batch_size=64, sampler=train_sampler,pin_memory=True,num_workers=8)
	max_tokens = args.max_tokens
	args.class_num = 3

	print("# args.img_fatures_size:", args.img_fatures_size)

	# Build the models
	encoder = ResNetFeats(args)
	decoder = models.setup(args)

	encoder.to(device)
	encoder.train(True)
	decoder.to(device)
	decoder.train(True)
	# NLLLoss
	nllloss = nn.NLLLoss().to(device)  # CrossEntropyLoss = log_softmax + NLLLoss
	# CenterLoss
	loss_weight = 1
	centerloss = CenterLoss(3, 10).to(device)

	# optimizer
	params = list(decoder.parameters())
	# optimizer
	if args.finetune_cnn:
		params += list(encoder.resnetLayer4.parameters())
	params += list(encoder.adaptive_pool7x7.parameters())
	# as in paper
	# d_model = embed_size
	# optimizer = NoamOpt(args.embed_size, 1, 4000,  \
	# 		torch.optim.Adam(params, lr=0, betas=(0.9, 0.98), eps=1e-9))

	optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.98), eps=1e-9)
	# optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9,weight_decay=0.01)

	start_epoch = 0
	# Train the models
	bestscore = 0# 2.860 #2.690
	loss_train = 0
	total_step = len(train_loader)
	iteration = start_epoch * total_step
	bestiter = iteration
	train_start = time.time()
	# args.num_epochs  = 0
	best_acc = 0
	best_epoch = 0
	for epoch in range(start_epoch, args.num_epochs):
		print("\n==>Epoch:", epoch)
		running_loss = 0.0
		running_correct = 0
		count = 0
		for i, current_batch in enumerate(tqdm(train_loader)):
			# npy_images = np.random.randint(0, 256, size=(64, 6, 224, 224)).astype(np.float32)
			# images = torch.from_numpy(npy_images)
			# true images
			element_img, all_ids, ele_ids, label = current_batch
			# images = raw_img
			# patch_images = element_img
			targets = label
			# images, targets,trans_tgt_masks,
			images = element_img.to(device)
			# patch_images = patch_images.to(device)
			targets = targets.to(device)
			# ResTrans --start--
			features = encoder(images)#[64,49,2408]
			# patch_features = encoder(patch_images)
			ip1, outputs = decoder(features)#[64,14,1475]
			# ResTrans --end--
			_, pred = torch.max(outputs.data, 1)
			# loss
			loss = nllloss(outputs, targets)
			# loss = nllloss(outputs, targets) + loss_weight * centerloss(targets, ip1)
			writer.add_scalar("Loss/train", loss, iteration)
			decoder.zero_grad()
			encoder.zero_grad()
			loss.backward()
			optimizer.step()
			iteration += 1
			count += 1
			# loss
			running_loss += loss
			# running_loss += loss.data[0]
			# accuracy
			running_correct += torch.sum(pred == targets.data)
		print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%".format(running_loss / (count*64),100 * running_correct / (count*64)))
		test_acc = test(args, "test", encoder=encoder, decoder=decoder)
		if test_acc>= best_acc:
			best_acc = test_acc
			best_epoch = epoch
			print("[INFO] save model")
			save_path = os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1))
			optim_state_dict = optimizer.state_dict()
			torch.save({'epoch': epoch,
						'decoder_state_dict': decoder.state_dict(),
						'encoder_state_dict': encoder.state_dict(),
						'optimizer': optim_state_dict,
						'iteration': iteration,
						args.score_select: test_acc,
						}, save_path)
			bestiter = iteration
		if test_acc> best_acc:
			bestscore = test_acc
			print(('[DEBUG] Saving model at epoch %d with %s score of %f' \
				   % (epoch, args.score_select, test_acc)))
			bestmodel_path = os.path.join(args.model_path, 'best_model.ckpt')
			os.system('cp %s %s' % (save_path, bestmodel_path))
	test_acc = test(args, "test", encoder=encoder, decoder=decoder)
	if test_acc >= best_acc:
		best_acc = test_acc
		best_epoch = epoch
		print("[INFO] save model")
		save_path = os.path.join(args.model_path, 'model-{}-{}.ckpt'.format(epoch + 1, i + 1))
		optim_state_dict = optimizer.state_dict()
		torch.save({'epoch': epoch,
					'decoder_state_dict': decoder.state_dict(),
					'encoder_state_dict': encoder.state_dict(),
					'optimizer': optim_state_dict,
					'iteration': iteration,
					args.score_select: test_acc,
					}, save_path)
		bestiter = iteration
	if test_acc > best_acc:
		bestscore = test_acc
		print(('[DEBUG] Saving model at epoch %d with %s score of %f' \
			   % (epoch, args.score_select, test_acc)))
		bestmodel_path = os.path.join(args.model_path, 'best_model.ckpt')
		os.system('cp %s %s' % (save_path, bestmodel_path))
	# train finished
	print('finished')


if __name__ == '__main__':
	args = get_opt()
	print(args)
	main(args)
