"""
This script selects incorrectly predicted images from the validation set and records them in record.txt
"""
import os
import json
import argparse
import sys
from functools import partial

import torch
from torchvision import transforms
from tqdm import tqdm

from my_dataset import MyDataSet
from model import BiFPN50 as create_model
from utils import seed_everything, worker_init_fn



def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    seed_everything(seed)
    OCT_class = ['AMD','CNV','CSR','DME','DR','DRUSEN','MH','NORMAL']
    OCT_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(OCT_class))

    root = '/hy-tmp/RetinalOCT_Dataset'
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    test_images_path = []
    test_images_label = []

    test_folder_path = os.path.join(root, "test")
    for current_dir, dirs, files in os.walk(test_folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported):
                test_images_path.append(os.path.join(current_dir, file))

    subdirectories_test = [name for name in os.listdir(test_folder_path) if os.path.isdir(os.path.join(test_folder_path, name))]
    for cla in subdirectories_test:
        image_class = class_indices[cla]
        cla_belong_path = os.path.join(test_folder_path, cla)
        for img_path in os.listdir(cla_belong_path):
            test_images_label.append(image_class)

    img_size = 224
    data_transform = {
        "test": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.2100, 0.2100, 0.2100], [0.1607, 0.1607, 0.1607])])}

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn,
                                              worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    model = create_model(num_classes=args.num_classes).to(device)

    assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    model.eval()
    with torch.no_grad():
        with open("record_incorrect.txt", "w") as f:
            data_loader = tqdm(test_loader, file=sys.stdout)
            for step, data in enumerate(data_loader):
                images, labels = data
                pred = model(images.to(device))
                pred_classes = torch.max(pred, dim=1)[1]
                contrast = torch.eq(pred_classes, labels.to(device)).tolist()
                labels = labels.tolist()
                pred_classes = pred_classes.tolist()
                for i, flag in enumerate(contrast):
                    if flag is False:
                        file_name = test_images_path[batch_size * step + i]
                        true_label = class_indict[str(labels[i])]
                        false_label = class_indict[str(pred_classes[i])]
                        f.write(f"{file_name}  TrueLabel:{true_label}  PredictLabel:{false_label}\n")


if __name__ == '__main__':
    seed = 42
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--weights', type=str, default="weights/best_model.pth",
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
