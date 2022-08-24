import random
import glob
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


class UVA(Dataset):
    def __init__(self, data_path, data_transform=None, length=32):
        self.data_path = data_path
        self.data_transform = data_transform
        self.length = length

    def load_video(self, file_name):
        video_path = self.data_path + '/' + file_name
        frames = sorted(glob.glob(video_path[:-4] + '_aligned/*.bmp'))

        step = len(frames)/self.length
        idxs = [round(i*step) for i in range(self.length)]

        video = [Image.open(frames[i]).convert('RGB') for i in idxs]

        if self.data_transform is not None:
            video = [self.data_transform(img) for img in video]
            video = torch.stack(video, 0).permute(1, 0, 2, 3)

        return video


class UVAage(UVA):
    def __init__(self, data_path, label, data_transform=None):
        super().__init__(data_path, data_transform)
        self.label = label

    @staticmethod
    def load_age(path):
        f = open(path, 'r')
        lines = f.readlines()[5:]
        pairs = []
        for line in lines:
            details = line.split('\t')
            filename = details[0]
            age = details[3]
            pairs.append((filename, age))
        return pairs

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        file_name, age = self.label[index]
        video = self.load_video(file_name)
        age = int(age)
        return video, age


class UVAtriplet(UVA):
    def __init__(self, data_path, label, data_transform=None):
        super().__init__(data_path, data_transform)
        self.label = label

    @staticmethod
    def read_file(path):
        f = open(path, 'r')
        lines = f.readlines()[5:]
        return lines

    @staticmethod
    def load_label(file_detail, kin_label, train=True):
        lines = file_detail
        file2code = {}
        code2file = defaultdict(list)
        for line in lines:
            details = line.split('\t')
            filename = details[0]
            code = details[1]
            file2code[filename] = code
            code2file[code].append(filename)

        lines = kin_label
        kindic1 = defaultdict(list)
        kindic2 = defaultdict(list)
        for line in lines:
            kin1 = line[0:3]
            kin2 = line[4:7]
            kindic1[kin1].append(kin2)
            kindic2[kin2].append(kin1)
        kindic = defaultdict(list)
        kindic.update(kindic1)
        kindic.update(kindic2)

        label = []
        for file, code in file2code.items():
            if len(kindic[code])>0:
                argu = 4 if train else 1
                for k in range(argu):
                    pos_code = random.sample(kindic[code], 1)[0]
                    pos = random.sample(code2file[pos_code], 1)[0]
                    neg_list = list(code2file.keys())
                    neg_list.remove(pos_code)
                    neg_code = random.sample(neg_list, 1)[0]
                    neg = random.sample(code2file[neg_code], 1)[0]
                    label.append((file, pos, neg))
        return label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)
        return anchor, pos, neg


if __name__ == '__main__':

    file_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_File_Details.txt'
    kin_label = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Kinship_Labels.txt'
    label = UVAtriplet.load_label(file_detail, kin_label)

    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

    import torchvision.transforms as transforms
    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = UVAtriplet(data_path, label[:int(0.8*len(label))], transform)

    train_loader = DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=4)

    for data in train_loader:
        anchor, pos, neg = data
        print(anchor.shape)
