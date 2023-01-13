# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
font = FontProperties(fname="Time_New_Roman.ttf", size=14)

def getdata(data_loc):
    epoch_list = []
    train_loss_list = []
    test_loss_list = []
    acc_one_list = []
    acc_five_list = []
    with open(data_loc, "r") as f:
        for i in f.readlines():
            data_i = i.split("\t")
            epoch_i = float(data_i[0][11:])
            train_loss_i = float(data_i[1][11:])
            test_loss_i = float(data_i[2][10:])
            acc_one = float(data_i[3][5:])
            acc_five = float(data_i[4][5:])
            epoch_list.append(epoch_i)
            train_loss_list.append(train_loss_i)
            test_loss_list.append(test_loss_i)
            acc_one_list.append(acc_one)
            acc_five_list.append(acc_five)

        return epoch_list, train_loss_list, test_loss_list, acc_one_list, acc_five_list


if __name__ == "__main__":
    data_loc = r"temp.txt"
    data_loc_ = r'exp_results.txt'
    epoch_list, train_loss, test_loss, top1, top5 = getdata(data_loc)

    epoch_list1, train_loss2, test_loss3, top1_, top5_ = getdata(data_loc_)

    plt.plot(epoch_list, top1, label="Adam")
    plt.plot(epoch_list, top1_, label="SGD")

    plt.legend()
    plt.xticks(np.arange(0, 20, 1))
    plt.yticks(np.arange(0, 2, 1))
    plt.xlabel("Epoch", fontproperties=font)
    plt.ylabel("Top-1 Accuracy(%)", fontproperties=font)
    plt.title("Comparison of accuracy of different optimizers.",fontproperties=font)
    plt.savefig("optimizers.jpg")
    plt.show()
