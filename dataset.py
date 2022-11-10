import random

import numpy as np
import pandas as pd
from decord import VideoReader, cpu
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch


class UvA_NEMO(Dataset):
    def __init__(self, data_path, length=16):
        self.data_path = data_path
        self.length = length

    def load_video(self, file_path):
        file_path = self.data_path + '/' + file_path[:-4] + '_aligned.avi'
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
        videoreader.seek(0)
        vid_len = len(videoreader)
        indices = np.linspace(0, vid_len - 1, num=self.length)
        indices = np.clip(indices, 0, vid_len - 1).astype(np.int64)
        video = videoreader.get_batch(indices).asnumpy()
        return video

    def load_image(self, file_path):
        detail_path = self.data_path + '/' + file_path[:-4] + '.csv'
        detail = pd.read_csv(detail_path)
        idx = detail[' AU12_r'].idxmax()
        file_path = self.data_path + '/' + file_path[:-4] + '_aligned.avi'
        videoreader = VideoReader(file_path, num_threads=1, ctx=cpu(0))
        frame = videoreader[idx].asnumpy()
        return frame


class Pretrained(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a = self.label[index]
        anchor = self.load_video(a)
        anchor = self.transform(list(anchor), return_tensors="pt")
        anchor = anchor['pixel_values'].squeeze()
        return anchor


class Age(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.subject2details = np.load('protocols/subject2details.npy', allow_pickle=True).item()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a = self.label[index]
        age = self.subject2details[a[:3]]['age']/100
        age = torch.FloatTensor([age])
        anchor = self.load_video(a)
        anchor = self.transform(list(anchor), return_tensors="pt")
        anchor = anchor['pixel_values'].squeeze()
        return anchor, age


class Classification(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16, cnn=False):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.cnn = cnn

    def __len__(self):
        return len(self.label)

    def cat_videos(self, vid1, vid2):
        vid1 = self.transform(list(vid1), return_tensors="pt")
        vid1 = vid1['pixel_values'].squeeze()
        vid2 = self.transform(list(vid2), return_tensors="pt")
        vid2 = vid2['pixel_values'].squeeze()
        vid_cat = [vid1,vid2]
        random.shuffle(vid_cat)
        return vid_cat

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)

        if torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, pos)
            label = torch.LongTensor([1])
        elif torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, neg)
            label = torch.LongTensor([0])
        else:
            anchor = self.cat_videos(pos, neg)
            label = torch.LongTensor([0])
        return anchor, label


class ClassificationCat(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16, cnn=False):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.cnn = cnn

    def __len__(self):
        return len(self.label)

    def cat_videos(self, vid1, vid2):
        vid1 = self.transform(list(vid1), return_tensors="pt")
        vid1 = vid1['pixel_values'].squeeze()
        vid2 = self.transform(list(vid2), return_tensors="pt")
        vid2 = vid2['pixel_values'].squeeze()
        vid_cat = [vid1,vid2]
        random.shuffle(vid_cat)
        vid_cat = torch.cat(vid_cat, 1)
        return vid_cat

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)

        if torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, pos)
            label = torch.LongTensor([1])
        elif torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, neg)
            label = torch.LongTensor([0])
        else:
            anchor = self.cat_videos(pos, neg)
            label = torch.LongTensor([0])
        return anchor, label


class Triplet(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16, cnn=False):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform
        self.cnn = cnn

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)

        anchor = self.transform(list(anchor), return_tensors="pt")
        pos = self.transform(list(pos), return_tensors="pt")
        neg = self.transform(list(neg), return_tensors="pt")

        if self.cnn:
            anchor = anchor['pixel_values'].squeeze().permute(1, 0, 2, 3)
            pos = pos['pixel_values'].squeeze().permute(1, 0, 2, 3)
            neg = neg['pixel_values'].squeeze().permute(1, 0, 2, 3)
        else:
            anchor = anchor['pixel_values'].squeeze()
            pos = pos['pixel_values'].squeeze()
            neg = neg['pixel_values'].squeeze()

        return anchor, pos, neg


class Image_pairs(UvA_NEMO):
    def __init__(self, data_path, label, transform=None, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_image(a)
        pos = self.load_image(p)
        neg = self.load_image(n)

        anchor = self.transform(anchor, return_tensors="pt")
        pos = self.transform(pos, return_tensors="pt")
        neg = self.transform(neg, return_tensors="pt")

        anchor = anchor['pixel_values'].squeeze()
        pos = pos['pixel_values'].squeeze()
        neg = neg['pixel_values'].squeeze()

        return anchor, pos, neg


class ImageClassification(UvA_NEMO):
    def __init__(self, data_path, label, transform, length=16):
        super().__init__(data_path, length)
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def cat_videos(self, vid1, vid2):
        vid1 = self.transform(vid1, return_tensors="pt")
        vid1 = vid1['pixel_values'].squeeze()
        vid2 = self.transform(vid2, return_tensors="pt")
        vid2 = vid2['pixel_values'].squeeze()
        vid_cat = [vid1,vid2]
        random.shuffle(vid_cat)
        return vid_cat

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_image(a)
        pos = self.load_image(p)
        neg = self.load_image(n)

        if torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, pos)
            label = torch.LongTensor([1])
        elif torch.rand(1)<0.5:
            anchor = self.cat_videos(anchor, neg)
            label = torch.LongTensor([0])
        else:
            anchor = self.cat_videos(pos, neg)
            label = torch.LongTensor([0])
        return anchor, label


if __name__ == '__main__':
    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'
    fold = 1
    group = 'wholeset/spontaneous'
    train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
    dataset = Image_pairs(data_path, train_label)
    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, num_workers=1)
    for f in loader:
        print(f[0].shape)