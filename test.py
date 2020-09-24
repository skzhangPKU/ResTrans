import argparse, os, time, pickle, json
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from models.image_models import ResNetFeats
from opts import get_opt
from optim import NoamOpt, LabelSmoothing
import models
from dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms

MEAN_TWO = (0.523307, 0.522698, 0.521938)
STD_TWO = (0.108646, 0.107423, 0.110502)

def save_test_json(preds, resFile):
    print(('Writing %d predictions' % (len(preds))))
    json.dump(preds, open(resFile, 'w'))


def test(args, split, modelfn=None, decoder=None, encoder=None):
    """Runs test on split=val/test with checkpoint file modelfn or loaded model_*"""
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find model directory
    if args.caption_model not in args.model_path:
        args.model_path += "_" + args.caption_model
        if args.finetune_cnn:
            args.model_path += "_finetune"

    # Get the best model path
    if encoder == None:
        modelfn = os.path.join(args.model_path, 'best_model.ckpt')

    # Build data loader
    # dir_name = 'E:\\OSLAB\\LearnDependency\\data_bank\\'
    dir_name = '/mnt/data/zhangsk/LearnData/'
    file_name = 'test_one.txt'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN_TWO, STD_TWO), transforms.ToPILImage()])
    test_data = MyDataset(root=dir_name, datatxt=file_name, transform=transform,target_transform=transforms.ToTensor())
    test_loader = DataLoader(dataset=test_data, batch_size=64,pin_memory=True,num_workers=8)
    # data_loader = get_loader(args, vocab, split, shuffle=False)

    max_tokens = args.max_tokens
    num_batches = len(test_loader)
    args.class_num = 3
    print(('[DEBUG] Running inference on %s with %d batches' % (split.upper(), num_batches)))

    # Load model
    if modelfn is not None:
        print(('[INFO] Loading checkpoint %s' % modelfn))
        encoder = ResNetFeats(args)
        decoder = models.setup(args)
        encoder.cuda()
        decoder.cuda()

        checkpoint = torch.load(modelfn)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])

    encoder.eval()
    decoder.eval()

    pred_captions = []
    running_correct = 0
    count = 0
    running_loss = 0.0
    for i, current_batch in enumerate(tqdm(test_loader)):
        # images, captions, _, _, _, img_ids, _ = current_batch
        element_img, all_ids, ele_ids, label = current_batch
        targets = label
        # images, targets,trans_tgt_masks,
        images = element_img.to(device)
        # patch_images = patch_images.to(device)
        targets = targets.to(device)

        images = images.to(device)
        features = encoder(images)
        ip1, outputs = decoder(features)
        _, pred = torch.max(outputs.data, 1)
        # print(pred)
        # loss = nllloss(outputs, targets)
        # loss = nllloss(outputs, targets) + centerloss(targets, ip1)
        # running_loss += loss
        # running_loss += loss.data[0]
        count = count + 1
        running_correct += torch.sum(pred == targets.data)
    test_acc = 100 * running_correct / (count * 64)
    # print("Test Loss is:{:.4f}, Test Accuracy is:{:.4f}%".format(running_loss / (count * 64), test_acc))
    print("Test Accuracy is:{:.4f}%".format(test_acc))

    encoder.train()
    decoder.train()

    return test_acc


def main(args):
    split = args.split
    test(args, split)


if __name__ == "__main__":
    args = get_opt()
    main(args)
