import os
import sys
import json
import pickle
import random

import numpy as np
import torch
from tqdm import tqdm

import matplotlib.pyplot as plt


def read_split_data(root: str):
    random.seed(42)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    OCT_class = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
    OCT_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(OCT_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]

    train_folder_path = os.path.join(root, "train")
    val_folder_path = os.path.join(root, "val")

    for current_dir, dirs, files in os.walk(train_folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported):
                train_images_path.append(os.path.join(current_dir, file))
    every_class_num.append(len(train_images_path))

    for current_dir, dirs, files in os.walk(val_folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported):
                val_images_path.append(os.path.join(current_dir, file))
    every_class_num.append(len(val_images_path))

    subdirectories_train = [name for name in os.listdir(train_folder_path)
                            if os.path.isdir(os.path.join(train_folder_path, name))]
    for cla in subdirectories_train:
        image_class = class_indices[cla]
        cla_belong_path = os.path.join(train_folder_path, cla)
        for img_path in os.listdir(cla_belong_path):
            train_images_label.append(image_class)

    subdirectories_val = [name for name in os.listdir(val_folder_path)
                          if os.path.isdir(os.path.join(val_folder_path, name))]
    for cla in subdirectories_val:
        image_class = class_indices[cla]
        cla_belong_path = os.path.join(val_folder_path, cla)
        for img_path in os.listdir(cla_belong_path):
            val_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = False
    if plot_image:
        plt.bar(range(len(OCT_class)), every_class_num, align='center')
        plt.xticks(range(len(OCT_class)), OCT_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = 'class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.4f}, acc: {:.4f}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def test(model, data_loader, device):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[test] loss: {:.4f}, acc: {:.4f}".format(
            accu_loss.item() / (step + 1), accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
