import os
import argparse
from functools import partial

import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

from my_dataset import MyDataSet
from model import MSR_TA_Net as create_model
from utils import read_split_data, train_one_epoch, evaluate, seed_everything, worker_init_fn


def main(args):
    seed_everything(seed=42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("weights") is False:
        os.makedirs("weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224

    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.2100, 0.2100, 0.2100], [0.1607, 0.1607, 0.1607])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.2100, 0.2100, 0.2100], [0.1607, 0.1607, 0.1607])
        ])
    }

    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = 8
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn,
                                               worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn,
                                             worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        pretrained_dict = torch.load(args.weights, map_location='cpu')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "classifier" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(pg, lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    train_losses = []
    train_accies = []
    val_losses = []
    val_accies = []
    lr_hist = []

    best_acc = 0.
    lowest_val_loss = float('inf')

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        train_losses.append(train_loss)
        train_accies.append(train_acc)
        val_losses.append(val_loss)
        val_accies.append(val_acc)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/best_model.pth")

        if val_loss <= lowest_val_loss:
            lowest_val_loss = val_loss
            torch.save(model.state_dict(), "weights/lowest_model.pth")

        torch.save(model.state_dict(), "weights/latest_model.pth")

        current_lr = optimizer.param_groups[0]["lr"]
        lr_hist.append(current_lr)
        scheduler.step()

    plt.figure()
    plt.plot(range(args.epochs), train_losses, label='Train Loss')
    plt.plot(range(args.epochs), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.show()

    plt.figure()
    plt.plot(range(args.epochs), train_accies, label='Train Accuracy')
    plt.plot(range(args.epochs), val_accies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    plt.show()

    plt.figure()
    plt.plot(range(args.epochs), lr_hist, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.savefig('learning_rate_schedule.png')
    plt.show()


if __name__ == '__main__':
    seed = 42
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--data-path', type=str, default=r"/mnt/RetinalOCT_Dataset")
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
