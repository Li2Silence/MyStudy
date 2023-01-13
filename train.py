# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from model import MyModel


def main():
    epochs = 20
    batch_size = 8
    save_path = 'best_model.pth' # 模型保存名字
    best_acc = 0.0
    image_path = r'data' # 数据集位置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 使用cuda训练模型

    """数据处理"""
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(128),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]),
        "test": transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])}

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    figure_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in figure_list.items())

    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"),
                                            transform=data_transform["test"])
    val_num = len(validate_dataset)

    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    net = MyModel(num_classes=36, init_weights=False)
    net.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.001, weight_decay=0.0005)

    train_steps = len(train_loader)
    test__steps = len(validate_loader)
    temp = []

    for epoch in range(epochs):
        # 模型训练
        net.train()
        running_loss = 0.0
        test__loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        # 计算top-1
        net.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss_ = loss_function(outputs, val_labels.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                test__loss += loss_.item()

        val_accurate = acc / val_num
        train_loss = running_loss / train_steps
        test_loss = test__loss / test__steps

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

        print('[epoch %d] train_loss: %.3f  top1_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 计算top-5
        net.eval()
        acc_ = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                maxk = max((1, 5))
                y_resize = val_labels.view(-1, 1)
                _, pred = outputs.topk(maxk, 1, True, True)
                acc_ += torch.eq(pred, y_resize.to(device)).sum().float().item()

        top5_accurate = acc_ / val_num

        data_ = "epoch_list:" + str(epoch) + '\t' + "train_loss:" + str('%.4f' % train_loss) + \
                '\t' + "test_loss:" + str('%.4f' % test_loss) + '\t' + 'top1:' + str('%.4f' % val_accurate) \
                + '\t' + "top5:" + str('%.4f' % top5_accurate) + '\n'

        temp.append(data_)

    with open('exp_results.txt', 'w', encoding='UTF-8') as f:
        for line in temp:
            f.write(str(line))

    print('Finished Training')


if __name__ == '__main__':
    main()
