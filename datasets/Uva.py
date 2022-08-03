import random
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import glob


class UVA(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.step = [2, 3]

        self.vids = os.listdir(self.data_path)
        self.transform = transform

    def __getitem__(self, idx):
        video_path = os.path.join(self.data_path, self.vids[idx])
        frames = sorted(glob.glob(video_path + '/*.jpg'))
        nframes = len(frames)
        step = random.sample(self.step, 1)[0]

        start_idx = random.randint(0, nframes - 16 * step)
        vid = [Image.open(frames[start_idx + i * step]).convert('RGB') for i in range(16)]

        if self.transform is not None:
            vid = self.transform(vid)

        return vid

    def __len__(self):
        return len(self.vids)


class UVAts(Dataset):
    def __init__(self, data_path, label, anchor_transform=None, posneg_transform=None):
        self.data_path = data_path
        self.step = [2, 3]

        self.label = label

        self.anchor_transform = anchor_transform
        self.posneg_transform = posneg_transform

    def load_video(self, idx):
        video_path = glob.glob(self.data_path + '/'+ idx + '*/')[0]
        frames = sorted(glob.glob(video_path + '/*.bmp'))
        nframes = len(frames)
        step = random.sample(self.step, 1)[0]

        start_idx = random.randint(0, nframes - 16 * step)
        # video = [Image.open(frames[start_idx + i * step]).convert('RGB') for i in range(16)]
        image = Image.open(frames[start_idx + step]).convert('RGB')

        return image

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        anchor_idx = list(self.label.keys())[index]
        target = anchor_idx
        anchor = self.load_video(anchor_idx)
        if self.anchor_transform is not None:
            anchor = self.anchor_transform(anchor)

        # now pair this up with an image from the same class in the second stream
        posneg_idx = self.label[anchor_idx]
        posneg = self.load_video(posneg_idx)

        if self.posneg_transform is not None:
            posneg = self.posneg_transform(posneg)
        return anchor, posneg, target


def load_label(label_path, split=1., shuff=False):

    f = open(label_path, 'r')
    lines = f.readlines()[5:]
    if shuff:
        random.shuffle(lines)
    train_label = {}
    val_label = {}
    for i in range(0, int((len(lines)) * split)):
        train_label[lines[i][0:3]] = lines[i][4:7]
    for i in range(int((len(lines)) * split), len(lines)):
        val_label[lines[i][0:3]] = lines[i][4:7]

    return train_label, val_label


if __name__ == '__main__':
    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'
    label_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Kinship_Labels.txt'

    label, _ = load_label(label_path, 0.8, True)
    print(label)

    dataset = UVAts(data_path, label)
    for i in range(len(dataset)):
        anchor, posneg, target = dataset.__getitem__(i)
        print(target)