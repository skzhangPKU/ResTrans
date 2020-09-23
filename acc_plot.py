# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def parser_log(logfile):
    train_accuracy = [[], []]
    test_accuracy = [[], []]
    with open(logfile, "r") as fp:
        for line in fp:
            tmp = line.split(' ')
            if tmp[0]=='==>Epoch:':
                epoch_index = int(tmp[-1].strip())
            if tmp[0]=='\n' or tmp[0]=='':
                continue
            if tmp[0]=='Loss':
                train_acc = float(tmp[-1].strip().split(':')[-1].replace('%', '')) / 100
                train_accuracy[0].append(epoch_index)
                train_accuracy[1].append(train_acc)
            if tmp[0] == 'Test':
                test_acc = float(tmp[-1].strip().split(':')[-1].replace('%', '')) / 100
                test_accuracy[0].append(epoch_index)
                test_accuracy[1].append(test_acc)
    return train_accuracy,test_accuracy




def py_plot(logfile):
    train_accuracy, test_accuracy = parser_log(logfile)

    plt.figure(1)
    plt.title("accuracy vs epoch")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.plot(test_accuracy[0], test_accuracy[1], 'b-', label='test accuracy')
    plt.plot(train_accuracy[0], train_accuracy[1], 'r-', label='train accuracy')
    plt.legend()
    plt.show()
    # plt.savefig(r'D:\accuracy.png')

if __name__ == '__main__':
    logfile = r"F:\\Project\\PycharmProj\\Baselines\\Oslab\\CNN_Dependency_Local\\plot.txt"
    py_plot(logfile)