import os

import numpy as np
import torch
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEFeatureExtractor, AutoFeatureExtractor, ResNetConfig

from KinshipVerificationModels import VideoMAEForKinshipVerification, SimpleCNNForKinshipVerification, \
    ResNet50forKinshipVerification
from dataset import Triplet, Image_pairs
from util import verification

device = torch.device('cuda')

data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=112)
image_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")


# def eval_image(model, val_data):
#     model.to(device)
#
#     model.eval()
#
#     embeddings1 = []
#     embeddings2 = []
#     is_same = []
#     with torch.no_grad():
#         with tqdm(val_data, unit="batch") as vepoch:
#             for batch in vepoch:
#                 batch = [v.to(device) for v in batch]
#                 out = model(batch)['logits']
#                 anchor_out = out[0].cpu().detach().numpy()
#                 pos_out = out[1].cpu().detach().numpy()
#                 neg_out = out[2].cpu().detach().numpy()
#                 embeddings1.append(anchor_out[0])
#                 embeddings2.append(pos_out[0])
#                 is_same.append(1)
#                 embeddings1.append(anchor_out[0])
#                 embeddings2.append(neg_out[0])
#                 is_same.append(0)
#         embeddings1 = np.array(embeddings1)
#         embeddings2 = np.array(embeddings2)
#         is_same = np.array(is_same)
#
#         diff = np.subtract(embeddings1, embeddings2)
#         dist = np.sum(np.square(diff), 1)
#         print('test:',dist)
#         best_threshold = 0.5
#         _, __, accuracy = verification.calculate_accuracy(best_threshold, dist, is_same)
#         print('Val Acc {}'.format(accuracy))


def evaluation(model, test_data, thresh):
    model.to(device)

    model.eval()

    embeddings1 = []
    embeddings2 = []
    is_same = []

    with torch.no_grad():
        with tqdm(test_data, unit="batch") as vepoch:
            for batch in vepoch:
                batch = [v.to(device) for v in batch]

                out = model(batch)['logits']

                anchor_out = out[0].cpu().detach().numpy()
                pos_out = out[1].cpu().detach().numpy()
                neg_out = out[2].cpu().detach().numpy()

                embeddings1.append(anchor_out[0])
                embeddings2.append(pos_out[0])
                is_same.append(1)
                embeddings1.append(anchor_out[0])
                embeddings2.append(neg_out[0])
                is_same.append(0)

    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    is_same = np.array(is_same)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    _, __, accuracy = verification.calculate_accuracy(thresh, dist, is_same)
    print('Acc {}, Used Thresh {}'.format(accuracy, thresh))

    return accuracy, len(test_data)


def image_evaluation(model, test_data, thresh):
    model.to(device)

    model.eval()

    embeddings1 = []
    embeddings2 = []
    is_same = []

    with torch.no_grad():
        with tqdm(test_data, unit="batch") as vepoch:
            for batch in vepoch:
                batch = [v.to(device) for v in batch]

                out = model(batch)['logits']

                anchor_out = out[0].cpu().detach().numpy()
                pos_out = out[1].cpu().detach().numpy()
                neg_out = out[2].cpu().detach().numpy()

                embeddings1.append(anchor_out[0])
                embeddings2.append(pos_out[0])
                is_same.append(1)
                embeddings1.append(anchor_out[0])
                embeddings2.append(neg_out[0])
                is_same.append(0)

    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    is_same = np.array(is_same)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    _, __, accuracy = verification.calculate_accuracy(thresh, dist, is_same)
    print('Acc {}, Used Thresh {}'.format(accuracy, thresh))

    return accuracy, len(test_data)


if __name__ == '__main__':
    # group = 'wholeset/spontaneous'
    # total_a = 0
    # total_l = 0
    # # for fold in range(1,6):
    # #     model = VideoMAEForKinshipVerification.from_pretrained(
    # #         'C:/Users/13661/PycharmProjects/ViViTforKin/checkpoints/' + group + '/vit/' + str(fold) + '/.',
    # #         local_files_only=True)
    # #
    # #     test_label = np.load('./protocols/' + group + '/test/' + str(fold) + '.npy')
    # #     test_set = Triplet(data_path, test_label, feature_extractor)
    # #     test_data = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
    # #
    # #     thresh = np.load('./checkpoints/' + group + '/vit/' + str(fold)+'/best_threshold.npy')
    # #
    # #     acc, l = evaluation(model, test_data, thresh)
    # #     total_a += acc*l
    # #     total_l += l
    # # print(total_a/total_l)
    #
    # for fold in range(1,6):
    #     model = SimpleCNNForKinshipVerification.from_pretrained(
    #         'C:/Users/13661/PycharmProjects/ViViTforKin/checkpoints/' + group + '/scnn/' + str(fold) + '/.',
    #         local_files_only=True)
    #
    #     test_label = np.load('./protocols/' + group + '/test/' + str(fold) + '.npy')
    #     test_set = Triplet(data_path, test_label, feature_extractor, cnn=True)
    #     test_data = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)
    #
    #     thresh = np.load('./checkpoints/' + group + '/scnn/' + str(fold)+'/best_threshold.npy')
    #
    #     acc, l = evaluation(model, test_data, thresh)
    #     total_a += acc*l
    #     total_l += l
    # print(total_a/total_l)

    # subsets = {'B-B': 7, 'S-B': 12, 'S-S': 7, 'F-D': 9, 'F-S': 12, 'M-S': 12, 'M-D': 16}
    subsets = {'M-D': 16}

    for subset, length in subsets.items():
        group = 'subset/'+subset+'/spontaneous'
        total_a = 0
        total_l = 0
        for fold in range(1,length+1):
            config = ResNetConfig(image_size=112, embedding_size=512, hidden_size=512,
                                  backbone=InceptionResnetV1(pretrained='vggface2'), loss='contrastive')
            model = ResNet50forKinshipVerification(config)
            path = os.path.join(os.getcwd(), '/checkpoints/' + group + '/facenet/' + str(fold)+'/best.pth')
            state_dict = torch.load(path)
            model.load_state_dict(state_dict)

            test_label = np.load('./protocols/' + group + '/test/' + str(fold) + '.npy')
            test_set = Image_pairs(data_path, test_label, image_feature_extractor)
            test_data = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=1)

            acc, l = evaluation(model, test_data, 0.5)
            total_a += acc*l
            total_l += l
        print(subset, total_a / total_l)
