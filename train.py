import os

import numpy as np
import torch
from accelerate import Accelerator
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, get_scheduler, ResNetConfig
from transformers import VideoMAEFeatureExtractor, VideoMAEConfig, AutoFeatureExtractor

from KinshipVerificationModels import VideoMAEForKinshipVerification, SimpleCNNForKinshipVerification, \
    ResNet50forKinshipVerification, ImageKinshipClassification
from dataset import Triplet, Image_pairs, ImageClassification
from util import verification
from util.LossFunc import ContrastiveLoss

data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

device = torch.device('cuda')

default_args = {
    "output_dir": "tmp",
    "evaluation_strategy": "steps",
    "num_train_epochs": 1,
    "log_level": "error",
    "report_to": "none",
}

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    fp16=True,
    learning_rate=5e-4,
    **default_args,
)

accelerator = Accelerator(fp16=training_args.fp16)

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=112)
image_feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50")


def train(model, train_data, val_data, num_epochs, mode, fold):
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)

    model, optimizer, train_data = accelerator.prepare(model, optimizer, train_data)

    num_training_steps = num_epochs * len(train_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    best_acc = 0
    best_vloss = 10000
    for epoch in range(1, num_epochs + 1):
        print('Fold {} Epoch {}'.format(fold, epoch))
        model.train()
        # model.backbone.eval()
        running_loss = 0.0
        with tqdm(train_data, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch, start=1):
                loss = model(batch)['loss']
                loss = loss / training_args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item() * training_args.gradient_accumulation_steps)
        avg_loss = running_loss / len(train_data) * training_args.gradient_accumulation_steps

        embeddings1 = []
        embeddings2 = []
        is_same = []
        with torch.no_grad():
            model.eval()
            with tqdm(val_data, unit="batch") as vepoch:
                for batch in vepoch:
                    batch = [v.to(device) for v in batch]
                    out = model(batch)['logits']
                    loss_fct = ContrastiveLoss(margin=1)
                    val_loss = loss_fct(out[0], out[1], 0)+loss_fct(out[0], out[2], 1)
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

        # tpr, fpr, accuracys, best_thresholds = verification.evaluate(embeddings1, embeddings2, is_same)
        # accuracy = accuracys.mean()
        # best_threshold = best_thresholds.mean()
        # print('Loss {}, Val Acc {}, Val Thresh {}'.format(avg_loss, accuracy, best_threshold))

        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
        print('test:',dist)
        best_threshold = 0.5
        _, __, accuracy = verification.calculate_accuracy(best_threshold, dist, is_same)
        print('Loss {}, Val Loss {}, Val Acc {}'.format(avg_loss, val_loss, accuracy))

        # if epoch % 5 == 0:
        #     model.save_pretrained(
        #         './out/' + mode + '/' + str(fold) + '/epoch' + str(epoch) + '/.')
        #     np.save('./out/' + mode + '/' + str(fold) + '/epoch' + str(epoch) + '/best_threshold.npy',
        #             best_threshold)

        if accuracy >= best_acc and val_loss < best_vloss:
            best_acc = accuracy
            best_vloss = val_loss
            print(111)
            path = os.path.join(os.getcwd(), '/checkpoints/' + mode + '/' + str(fold))
            if not os.path.exists(path):
                os.makedirs(path)
            print(path)
            torch.save(model.state_dict(), path + '/best.pth')
            # model.save_pretrained('./checkpoints/' + mode + '/' + str(fold) + '/.')
            # np.save('./checkpoints/' + mode + '/' + str(fold) + '/best_threshold.npy', best_threshold)

        # if accuracy >= best_acc:
        #     best_acc = accuracy
        #     model.save_pretrained('./checkpoints/' + mode + '/' + str(fold) + '/.')
        #     np.save('./checkpoints/' + mode + '/' + str(fold) + '/best_threshold.npy', best_threshold)


def train_classifi(model, train_data, val_data, num_epochs, group, fold):
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)

    model, optimizer, train_data = accelerator.prepare(model, optimizer, train_data)

    num_training_steps = num_epochs * len(train_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    best_val_loss = 10
    for epoch in range(1, num_epochs + 1):
        print('Fold {} Epoch {}'.format(fold, epoch))
        model.train()
        running_loss = 0.0
        with tqdm(train_data, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch, start=1):
                outputs = model(batch[0], labels=batch[1])
                loss = outputs.loss / training_args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item() * training_args.gradient_accumulation_steps)
        avg_loss = running_loss / len(train_data) * training_args.gradient_accumulation_steps

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            with tqdm(val_data, unit="batch") as vepoch:
                for batch in vepoch:
                    batch[0] = [v.to(device) for v in batch[0]]
                    batch[1] = batch[1].to(device)
                    outputs = model(batch[0], labels=batch[1])
                    loss = outputs.loss
                    val_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_data)

        print('Loss training {} val loss {}'.format(avg_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # model.save_pretrained('./checkpoints/' + group + '/facenet_cla/' + str(fold) + '/.', state_dict=model.state_dict())


def train_wholeset_vit():
    group = 'wholeset/spontaneous'
    for fold in range(1, 6):
        train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
        train_set = Triplet(data_path, train_label, feature_extractor)
        train_data = DataLoader(
            train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
            pin_memory=True)

        val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
        val_set = Triplet(data_path, val_label, feature_extractor)
        val_data = DataLoader(
            val_set, batch_size=1, shuffle=True, num_workers=1)

        config = VideoMAEConfig(image_size=112, embedding_size=128, loss='triplet')
        model = VideoMAEForKinshipVerification.from_pretrained(
            'C:/Users/13661/PycharmProjects/ViViTforKin/pretrained/age/' + group + '/' + str(fold) + '/.',
            local_files_only=True, config=config)

        train(model, train_data, val_data, 5, group + '/vit', fold)


def train_wholeset_cnn():
    group = 'wholeset/spontaneous'
    for fold in range(1, 6):
        train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
        train_set = Triplet(data_path, train_label, feature_extractor, cnn=True)
        train_data = DataLoader(
            train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
            pin_memory=True)

        val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
        val_set = Triplet(data_path, val_label, feature_extractor, cnn=True)
        val_data = DataLoader(
            val_set, batch_size=1, shuffle=True, num_workers=1)

        config = VideoMAEConfig(image_size=112, embedding_size=128, loss='contrastive')
        model = SimpleCNNForKinshipVerification(config)

        train(model, train_data, val_data, 10, group + '/scnn', fold)


def train_subsets_cnn():
    subsets = {'M-D': 16}
    # subsets = {'B-B': 7,'S-B': 12, 'S-S': 7, 'F-D': 9, 'F-S': 12, 'M-S': 12, 'M-D': 16}

    for subset, length in subsets.items():
        group = 'subset/'+subset+'/spontaneous'
        for fold in range(1, length+1):
            train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
            train_set = Triplet(data_path, train_label, feature_extractor, cnn=True)
            train_data = DataLoader(
                train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=1,
                pin_memory=True)
            val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
            val_set = Triplet(data_path, val_label, feature_extractor, cnn=True)
            val_data = DataLoader(
                val_set, batch_size=1, shuffle=True, num_workers=1)

            config = VideoMAEConfig(image_size=112, embedding_size=128, loss='contrastive')
            model = SimpleCNNForKinshipVerification(config)

            train(model, train_data, val_data, 5, group + '/scnn', fold)


def train_subsets_vit():
    subsets = {'B-B': 7,'S-B': 12, 'S-S': 7, 'F-D': 9, 'F-S': 12, 'M-S': 12, 'M-D': 16}

    for subset, length in subsets.items():
        group = 'subset/'+subset+'/spontaneous'
        for fold in range(1, length+1):
            train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
            train_set = Triplet(data_path, train_label, feature_extractor)
            train_data = DataLoader(
                train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=1,
                pin_memory=True)
            val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
            val_set = Triplet(data_path, val_label, feature_extractor)
            val_data = DataLoader(
                val_set, batch_size=1, shuffle=True, num_workers=1)

            config = VideoMAEConfig(image_size=112, num_hidden_layers=4, embedding_size=128, loss='contrastive')
            model = VideoMAEForKinshipVerification.from_pretrained(
                'C:/Users/13661/PycharmProjects/ViViTforKin/pretrained/'+ 'wholeset/spontaneous'+ '/' + str(fold) + '/.',
                local_files_only=True, config=config)

            train(model, train_data, val_data, 5, group + '/vit', fold)


def train_subsets_resnet():
    subsets = {'B-B': 7,'S-B': 12, 'S-S': 7, 'F-D': 9, 'F-S': 12, 'M-S': 12, 'M-D': 16}
    # subsets = {'M-D': 16}

    for subset, length in subsets.items():
        group = 'subset/'+subset+'/spontaneous'
        for fold in range(1, length+1):
            train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
            train_set = Image_pairs(data_path, train_label, image_feature_extractor)
            train_data = DataLoader(
                train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=1,
                pin_memory=True)
            val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
            val_set = Image_pairs(data_path, val_label, image_feature_extractor)
            val_data = DataLoader(
                val_set, batch_size=1, shuffle=True, num_workers=1)

            config = ResNetConfig(image_size=112, embedding_size=512, hidden_size = 512, backbone=InceptionResnetV1(pretrained='vggface2'), loss='contrastive')
            model = ResNet50forKinshipVerification(config)
            # model.resnet.from_pretrained("microsoft/resnet-50")
            # for param in model.backbone.parameters():
            #     param.requires_grad = False
            # i = 0
            # for child in model.backbone.children():
            #     for c in child.children():
            #         i += 1
            #         if i >= 3:
            #             for param in c.parameters():
            #                 param.requires_grad = True

            train(model, train_data, val_data, 5, group + '/facenet_full', fold)


def train_image_classifi():
    group = 'wholeset/spontaneous'
    for fold in range(1, 6):
        config = VideoMAEConfig(image_size=112, num_labels=2, hidden_size=512, backbone=InceptionResnetV1(pretrained='vggface2'))
        model = ImageKinshipClassification(config=config)
        for param in model.backbone.parameters():
            param.requires_grad = False

        train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
        train_set = ImageClassification(data_path, train_label, image_feature_extractor)
        train_data = DataLoader(
            train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
            pin_memory=True)

        val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
        val_set = ImageClassification(data_path, val_label, image_feature_extractor)
        val_data = DataLoader(
            val_set, batch_size=1, shuffle=True, num_workers=4,
            pin_memory=True)

        train_classifi(model, train_data, val_data, 20, group, fold)


if __name__ == '__main__':
    train_subsets_resnet()
    # train_wholeset_vit()
    # train_image_classifi()
