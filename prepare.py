import os
import random
from collections import defaultdict
from itertools import product


def read_file(path):
    f = open(path, 'r')
    lines = f.readlines()[5:]
    return lines


subject_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Subject_Details.txt'
file_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_File_Details.txt'
kin_label = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Kinship_Labels.txt'

discard_attr = ['granddaughter,grandmother', 'granddaughter,grandfather', 'grandfather,grandson']
# + ['son,mother','daughter,mother', 'father,daughter', 'son,father', 'father,son', 'daughter,father', 'mother,daughter', 'mother,son']

subsets2attr = {'B-B': ['brother,brother'], 'S-B': ['sister,brother', 'brother,sister'], 'S-S': ['sister,sister'],
                'F-D': ['daughter,father', 'father,daughter'], 'F-S': ['son,father', 'father,son'],
                'M-S': ['mother,son', 'son,mother'], 'M-D': ['daughter,mother', 'mother,daughter']}

pairs = []
pairs_by_attr = defaultdict(list)

lines = read_file(kin_label)
for line in lines:
    kin1 = line[0:3]
    kin2 = line[4:7]
    relation = line.split('\t')[1][:-1]
    if relation not in discard_attr:
        pairs.append({kin1, kin2})
        pairs_by_attr[relation].append((kin1, kin2))

subject_codes = list(set([s for p in pairs for s in p]))

print(subject_codes)


subject2files = defaultdict(lambda: {'deliberate': [], 'spontaneous': []})
lines = read_file(file_detail)
for line in lines:
    details = line.split('\t')
    filename = details[0]
    smile_type = filename.split('_')[1]
    code = details[1]
    subject2files[code][smile_type].append(filename)

print(subject2files)


pairs_by_subsets = defaultdict(list)
for label, attr in subsets2attr.items():
    for a in attr:
        pairs_by_subsets[label] += pairs_by_attr[a]

print(pairs_by_subsets)

# subsets = {}
# for subset, pairs in pairs_by_subsets.items():
#     subsets[subset] = len(pairs)
# print(subsets)


subject2details = {}

lines = read_file(subject_detail)

for line in lines:
    details = line.split('\t')
    subject = {'gender': details[1].strip(), 'age': int(details[2])}
    subject2details[details[0]] = subject

print(subject2details)

family = []
pairs_in_family = []
for p in pairs:
    if len(family) == 0:
        family.append(p)
        pairs_in_family.append([p])
        continue
    flag = False
    for i, f in enumerate(family):
        if len(f.intersection(p)) > 0:
            family[i] = f.union(p)
            pairs_in_family[i].append(p)
            flag = True
            continue
    if flag:
        continue
    else:
        family.append(p)
        pairs_in_family.append([p])

print(family)
print(pairs_in_family)


def pair_triplet_family(indexs, smile_type, train=True):
    triplets = []

    train_subjects = set([s for i in indexs for s in family[i]])

    print(len(train_subjects), train_subjects)

    for index in indexs:
        f = family[index]
        pos_pairs = pairs_in_family[index]
        neg_list = list(train_subjects.difference(f))
        for pos_pair in pos_pairs:
            pos_pair = list(pos_pair)
            anchor = subject2files[pos_pair[0]][smile_type]
            pos = subject2files[pos_pair[1]][smile_type]
            pos_age = subject2details[pos_pair[1]]['age']
            pos_gender = subject2details[pos_pair[1]]['gender']
            neg_codes = []
            for neg in neg_list:
                if len(subject2files[neg][smile_type]) > 0 and subject2details[neg]['gender'] == pos_gender and \
                        (pos_age - 10) < subject2details[neg]['age'] < (pos_age + 10):
                    neg_codes.append(neg)
            if len(anchor) * len(pos) * len(neg_codes) == 0:
                continue
            if train:
                pos_pair_files = list(product(*[anchor, pos]))
            else:
                pos_pair_files = [[anchor[0], pos[0]]]

            for pos_pair_file in pos_pair_files:
                pos_pair_file = list(pos_pair_file)
                if train:
                    neg_files = [neg_file for neg in neg_codes for neg_file in subject2files[neg][smile_type]]
                    if len(neg_files) == 0:
                        continue
                    neg_files = random.sample(neg_files, min(len(neg_files), 4))
                    for neg_file in neg_files:
                        trip = [pos_pair_file[0], pos_pair_file[1], neg_file]
                        triplets.append(trip)
                else:
                    neg = random.choice(neg_codes)
                    neg = random.choice(subject2files[neg][smile_type])
                    pos_pair_file.append(neg)
                    triplets.append(pos_pair_file)

    return triplets


def pair_triplet(subsets, train_index, smile_type, train=True, skip_codes=()):
    triplets = []

    train_subjects = set([s for i in train_index for s in subsets[i]])
    print(train_subjects)

    for index in train_index:
        pos_pair = subsets[index]
        pos_pair = list(pos_pair)
        random.shuffle(pos_pair)
        neg_list = set(subject_codes).difference(pos_pair)
        neg_list = list(neg_list.difference(skip_codes))
        anchor = subject2files[pos_pair[0]][smile_type]
        pos = subject2files[pos_pair[1]][smile_type]
        pos_age = subject2details[pos_pair[1]]['age']
        pos_gender = subject2details[pos_pair[1]]['gender']
        neg_codes = []
        for neg in neg_list:
            if len(subject2files[neg][smile_type]) > 0 and subject2details[neg]['gender'] == pos_gender and \
                    (pos_age - 10) < subject2details[neg]['age'] < (pos_age + 10):
                neg_codes.append(neg)
        if len(anchor) * len(pos) * len(neg_codes) == 0:
            print(pos_pair, skip_codes, len(anchor), len(pos), len(neg_codes))
            continue
        if train:
            pos_pair_files = list(product(*[anchor, pos]))
        else:
            pos_pair_files = [[anchor[0], pos[0]]]

        for pos_pair_file in pos_pair_files:
            pos_pair_file = list(pos_pair_file)
            if train:
                neg_files = [neg_file for neg in neg_codes for neg_file in subject2files[neg][smile_type]]
                if len(neg_files) == 0:
                    print(pos_pair, neg_codes)
                    continue
                neg_files = random.sample(neg_files, min(len(neg_files), 4))
                for neg_file in neg_files:
                    trip = [pos_pair_file[0], pos_pair_file[1], neg_file]
                    triplets.append(trip)
            else:
                neg = random.choice(neg_codes)
                neg_file = random.choice(subject2files[neg][smile_type])
                pos_pair_file.append(neg_file)
                triplets.append(pos_pair_file)
                return triplets, neg
    return triplets


if __name__ == '__main__':
    import numpy as np
    from sklearn.model_selection import KFold

    np.save('protocols/family.npy', family)
    np.save('protocols/subject2details.npy', subject2details)

    smile_type = 'spontaneous'
    kf = KFold(n_splits=5, shuffle=True)
    fold = 0
    # for train_index, test_index in kf.split(family):
    #     train_index = list(train_index)
    #     test_index = list(test_index)
    #     random.shuffle(train_index)
    #     train = train_index[:-1*len(test_index)]
    #     val = train_index[-1 * len(test_index):]
    #     print("TRAIN:", train, "\nVAL:", val, "\nTEST:", test_index)
    #     X_train, X_val, X_test = pair_triplet_family(train, smile_type), pair_triplet_family(val, smile_type, False), pair_triplet_family(test_index, smile_type, False)
    #     fold += 1
    #     random.shuffle(X_train)
    #     print(X_train)
    #     print(len(X_train))
    #     np.save('protocols/wholeset/' + smile_type + '/train/' + str(fold) + '.npy', X_train)
    #     random.shuffle(X_val)
    #     print(X_val)
    #     print(len(X_val))
    #     np.save('protocols/wholeset/' + smile_type + '/val/' + str(fold) + '.npy', X_val)
    #     random.shuffle(X_test)
    #     print(X_test)
    #     print(len(X_test))
    #     np.save('protocols/wholeset/' + smile_type + '/test/' + str(fold) + '.npy', X_test)
    #
    # for subset, raw_pairs in pairs_by_subsets.items():
    #     print(subset)
    #     # clean pairs
    #     pairs = []
    #     for pair in raw_pairs:
    #         if len(subject2files[pair[0]][smile_type])==0 or len(subject2files[pair[1]][smile_type])==0:
    #             print(pair)
    #         else:
    #             pairs.append(pair)
    #
    #     kf = KFold(n_splits=len(pairs), shuffle=True)
    #     fold = 0
    #     for train_index, test_index in kf.split(pairs):
    #         fold += 1
    #         print(subset, fold)
    #         train_index = list(train_index)
    #         test_index = list(test_index)
    #         random.shuffle(train_index)
    #         train = train_index[:-1*len(test_index)]
    #         val = train_index[-1 * len(test_index):]
    #         print("TRAIN:", train, "\nVAL:", val, "\nTEST:", test_index)
    #
    #         train_subjects = set([s for i in train for s in pairs[i]])
    #         val_subjects = set([s for i in val for s in pairs[i]])
    #         test_subjects = set([s for i in test_index for s in pairs[i]])
    #
    #         X_val, neg_val = pair_triplet(pairs, val, smile_type, train=False, skip_codes=set.union(train_subjects, test_subjects))
    #         for f in family:
    #             if neg_val in f:
    #                 neg_val = set(f)
    #         X_test, neg_test = pair_triplet(pairs, test_index, smile_type, train=False, skip_codes=set.union(train_subjects, val_subjects, neg_val))
    #         for f in family:
    #             if neg_test in f:
    #                 neg_test = set(f)
    #         X_train = pair_triplet(pairs, train, smile_type, train=True, skip_codes=set.union(test_subjects, val_subjects, neg_test, neg_val))
    #         random.shuffle(X_train)
    #         print(X_train)
    #         print(len(X_train))
    #         path = os.path.join(os.getcwd(), 'protocols/subset/' + subset + '/' + smile_type + '/train')
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         np.save('protocols/subset/' + subset + '/' + smile_type+'/train/' + str(fold) + '.npy', X_train)
    #         random.shuffle(X_val)
    #         print(X_val)
    #         print(len(X_val))
    #         path = os.path.join(os.getcwd(), 'protocols/subset/' + subset + '/' + smile_type + '/val')
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         np.save('protocols/subset/' + subset + '/' + smile_type+'/val/' + str(fold) + '.npy', X_val)
    #         random.shuffle(X_test)
    #         print(X_test)
    #         print(len(X_test))
    #         path = os.path.join(os.getcwd(), 'protocols/subset/' + subset + '/' + smile_type + '/test')
    #         if not os.path.exists(path):
    #             os.makedirs(path)
    #         np.save('protocols/subset/' + subset + '/' + smile_type+'/test/' + str(fold) + '.npy', X_test)


