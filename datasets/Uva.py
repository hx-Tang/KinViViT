import random
import glob
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader


def load_subjects(subject_label, file_details, kinship_labels):
    subjects = defaultdict(None)

    lines = read_file(subject_label)

    for line in lines:
        details = line.split('\t')
        subject = {'gender': details[1].strip(), 'age': int(details[2]), 'kin': {}, 'files': []}
        subjects[details[0]] = subject

    lines = read_file(kinship_labels)
    for line in lines:
        kin1 = line[0:3]
        kin2 = line[4:7]
        relation = line.split('\t')[1][:-1]
        if kin2 not in subjects[kin1]['kin']:
            subjects[kin1]['kin'][kin2] = relation
        if kin1 not in subjects[kin2]['kin']:
            subjects[kin2]['kin'][kin2] = relation

    lines = read_file(file_details)
    for line in lines:
        details = line.split('\t')
        filename = details[0]
        code = details[1]
        subjects[code]['files'].append(filename)

    return subjects


def gen_triplets(subjects, keys, argu=1):
    labels = []
    j = 0
    for key in keys:
        pos_list = [k for k in subjects[key]['kin'].keys() if k in keys]
        if len(pos_list) == 0:
            continue
        elif len(pos_list) < argu:
            pos_codes = pos_list + [random.choice(pos_list) for _ in range(argu - len(pos_list))]
        else:
            pos_codes = random.sample(pos_list, argu)
        j += 1

        neg_list = keys[:]
        for k in list(subjects[key]['kin'].keys()) + [key]:
            if k in neg_list:
                neg_list.remove(k)
        if len(neg_list) < argu:
            neg_codes = neg_list + [random.choice(neg_list) for _ in range(argu - len(neg_list))]
        else:
            neg_codes = random.sample(neg_list, argu)

        for anchor in subjects[key]['files']:
            for i in range(argu):
                pos = random.sample(subjects[pos_codes[i]]['files'], 1)[0]
                neg = random.sample(subjects[neg_codes[i]]['files'], 1)[0]
                labels.append([anchor, pos, neg])

    return labels


def read_file(path):
    f = open(path, 'r')
    lines = f.readlines()[5:]
    return lines


class UVA(Dataset):
    def __init__(self, data_path, data_transform=None, length=16):
        self.data_path = data_path
        self.data_transform = data_transform
        self.length = length

    def read_file(self, path):
        f = open(path, 'r')
        lines = f.readlines()[5:]
        return lines

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


class UVAtriplet2(UVA):
    def __init__(self, data_path, label, data_transform=None):
        super().__init__(data_path, data_transform)
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        a, p, n = self.label[index]
        anchor = self.load_video(a)
        pos = self.load_video(p)
        neg = self.load_video(n)
        return anchor, pos, neg


if __name__ == '__main__':

    subject_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Subject_Details.txt'
    file_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_File_Details.txt'
    kin_label = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Kinship_Labels.txt'

    subjects = load_subjects(subject_detail, file_detail, kin_label)
    print(subjects)

    # print(len([v['kin'] for k, v in subjects.items() if len(v['kin'])>0]))

    subjects_code = list(subjects.keys())
    random.shuffle(subjects_code)
    import numpy as np

    for fold in range(5):
        length = len(subjects_code)
        # print(len(subjects_code))
        train_label = gen_triplets(subjects, subjects_code[:int(fold*length*0.2)]+subjects_code[int((fold+1)*length*0.2):], 4)
        random.shuffle(train_label)
        print(len(train_label))
        # print(len(subjects_code[:int(fold*length*0.2)]+subjects_code[int((fold+1)*length*0.2):]))
        # np.save('../fold_argu/train_label_fold'+str(fold+1)+'.npy', train_label)
        test_label = gen_triplets(subjects, subjects_code[int(fold*length*0.2):int((fold+1)*length*0.2)], 1)
        random.shuffle(test_label)
        print(len(test_label))
        # print(len(subjects_code[int(fold*length*0.2):int((fold+1)*length*0.2)]))
        # np.save('../fold_argu/test_label_fold'+str(fold+1)+'.npy', test_label)

    # data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'
    #
    # import torchvision.transforms as transforms
    # transform = transforms.Compose(
    #     [transforms.ToTensor()])
    #
    # trainset = UVAtriplet(data_path, label[:int(0.8*len(label))], transform)
    #
    # train_loader = DataLoader(
    #     trainset, batch_size=16, shuffle=True, num_workers=4)
    #
    # for data in train_loader:
    #     anchor, pos, neg = data
