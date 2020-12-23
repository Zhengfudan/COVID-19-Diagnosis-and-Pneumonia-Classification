import torchvision 
import torch.nn as nn
import torch
import time
import datetime
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from senet.se_resnet import se_resnet50
import seaborn as sns

parser.add_argument("--data_path", type=str, default='', help="visualization img path")
parser.add_argument("--checkpoints_path", type=str, default='checkpoints/Best_model.pth', help="nest model")
opt = parser.parse_args()

img_path = opt.data_path
checkpoints = opt.checkpoints_path

# functions to show an image
def getFeat(data, savepath):
    # data [h, w]
    f, ax = plt.subplots()
    sns.heatmap(data, ax=ax, cmap='jet', annot=False, cbar=False, xticklabels=False, yticklabels=False)
    # 去除边框
    height, width = data.shape[0], data.shape[1]
    # 如果dpi=300，那么图像大小=height*width
    f.set_size_inches(width / 100.0, height / 100.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.axis('off')
    plt.margins(0, 0)
    # 保存
    plt.savefig(savepath)
    plt.close()
def imshow(img, i, j):
    npimg = img.numpy()
    npimg -= npimg.min()
    npimg /= npimg.max()
    getFeat(npimg[0], f'viral/visual{i}_{j}.png')

class Xinguan(Dataset):
    def __init__(self, img_path_list):
        super(Xinguan, self).__init__()
        self.img_path_list = img_path_list
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        img = Image.open(img_path)
        img = img.resize((512, 512))
        img = self.preprocess(img)
        return {'image': torch.from_numpy(img).float()}

    def __len__(self): 
        return len(self.img_path_list)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = se_resnet50(num_classes=1)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(checkpoints))
    filename = os.listdir(img_path)
    img_path_list = []
    for i in filename:
        img_path_list.append(os.path.join(img_path, i))
    dataset = Xinguan(img_path_list)
    dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=0
        )
    images = next(iter(dataloader))['image'].to(device)

    images = model.module.conv1(images)
    images = model.module.bn1(images)
    images = model.module.relu(images)
    images = model.module.maxpool(images)
    images = model.module.layer1[0].conv1(images)
    images = model.module.layer1[0].bn1(images)
    images = model.module.layer1[0].conv2(images)
    images = model.module.layer1[0].bn2(images)
    images = model.module.layer1[0].conv3(images)
    images = model.module.layer1[0].bn3(images)
    images = model.module.layer1[0].relu(images)
    images = model.module.layer1[0].se(images)

    for i, image in enumerate(images):
        image2 = image.detach().cpu()
        for j, image3 in enumerate(image2):
            image4 = image3.unsqueeze(dim=0).unsqueeze(dim=0)
            imshow(torchvision.utils.make_grid(image4), i, j)
