import os
import json
from functools import partial

import torch
from torchvision import transforms
from tqdm import tqdm
from utils import test
from my_dataset import MyDataSet
from model import MSR_TA_Net as create_model
from confusion_matrix import ConfusionMatrix
from utils import seed_everything, worker_init_fn


def main():
    seed_everything(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    OCT_class = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
    OCT_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(OCT_class))

    root = '/mnt/RetinalOCT_Dataset'
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    test_images_path = []
    test_images_label = []

    test_folder_path = os.path.join(root, "test")
    for current_dir, dirs, files in os.walk(test_folder_path):
        for file in files:
            if any(file.endswith(ext) for ext in supported):
                test_images_path.append(os.path.join(current_dir, file))

    subdirectories_test = [name for name in os.listdir(test_folder_path)
                           if os.path.isdir(os.path.join(test_folder_path, name))]
    for cla in subdirectories_test:
        image_class = class_indices[cla]
        cla_belong_path = os.path.join(test_folder_path, cla)
        for img_path in os.listdir(cla_belong_path):
            test_images_label.append(image_class)

    print("{} images for testing.".format(len(test_images_path)))
    assert len(test_images_path) > 0, "number of testing images must greater than 0."

    img_size = 224
    data_transform = {
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.2100, 0.2100, 0.2100], [0.1607, 0.1607, 0.1607])
        ])
    }

    test_dataset = MyDataSet(images_path=test_images_path,
                             images_class=test_images_label,
                             transform=data_transform["test"])

    batch_size = 32
    nw = 8
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=nw,
                                              collate_fn=test_dataset.collate_fn,
                                              worker_init_fn=partial(worker_init_fn, rank=0, seed=seed))

    model = create_model(num_classes=8).to(device)

    model_weight_path = "weights/best_model.pth"
    assert os.path.exists(model_weight_path), "model weight file: '{}' does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()

    test_loss, test_acc = test(model=model, data_loader=test_loader, device=device)

    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=8, labels=labels)
    model.eval()
    with torch.no_grad():
        for test_data in tqdm(test_loader):
            test_images, test_labels = test_data
            outputs = model(test_images.to(device))
            outputs = torch.softmax(outputs, dim=1)
            outputs = torch.argmax(outputs, dim=1)
            confusion.update(outputs.to("cpu").numpy(), test_labels.to("cpu").numpy())
    confusion.plot()
    confusion.summary()


if __name__ == '__main__':
    seed = 42
    main()
