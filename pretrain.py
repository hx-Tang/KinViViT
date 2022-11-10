import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments, get_scheduler, VideoMAEForVideoClassification
from transformers import VideoMAEFeatureExtractor, VideoMAEForPreTraining, VideoMAEConfig

from KinshipVerificationModels import VideoMAEForKinshipClassification
from dataset import Pretrained, Age, Classification, ClassificationCat

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    fp16=True,
    learning_rate=5e-5,
    **default_args,
)

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base", size=112)


def pretrain(model, pretrain_data, num_epochs, group, fold):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)
    accelerator = Accelerator(fp16=training_args.fp16)
    model, optimizer, pretrain_data = accelerator.prepare(model, optimizer, pretrain_data)

    num_training_steps = num_epochs * len(pretrain_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (16 // model.config.tubelet_size) * num_patches_per_frame
    bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

    best_loss = 10
    for epoch in range(1, num_epochs + 1):
        print('Fold {} Epoch {}'.format(fold, epoch))
        model.train()
        running_loss = 0.0
        with tqdm(pretrain_data, unit="batch") as tepoch:
            for step, batch in enumerate(tepoch, start=1):
                outputs = model(batch, bool_masked_pos=bool_masked_pos)
                loss = outputs.loss / training_args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % training_args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                running_loss += loss.item()
                tepoch.set_postfix(loss=loss.item() * training_args.gradient_accumulation_steps)
        avg_loss = running_loss / len(pretrain_data) * training_args.gradient_accumulation_steps

        print('Loss pretrain {}'.format(avg_loss))

        if avg_loss < best_loss:
            best_loss = avg_loss
            model.save_pretrained('./pretrained/' + group + '/' + str(fold) + '/.')


def pretrain_age(model, age_data, val_data, num_epochs, group, fold):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)
    accelerator = Accelerator(fp16=training_args.fp16)
    model, optimizer, age_data = accelerator.prepare(model, optimizer, age_data)

    num_training_steps = num_epochs * len(age_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0,
                                 num_training_steps=num_training_steps)

    best_val_loss = 10
    for epoch in range(1, num_epochs + 1):
        print('Fold {} Epoch {}'.format(fold, epoch))
        model.train()
        running_loss = 0.0
        with tqdm(age_data, unit="batch") as tepoch:
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
        avg_loss = running_loss / len(age_data) * training_args.gradient_accumulation_steps

        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            with tqdm(val_data, unit="batch") as vepoch:
                for batch in vepoch:
                    batch = [v.to(device) for v in batch]
                    outputs = model(batch[0], labels=batch[1])
                    loss = outputs.loss
                    val_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_data)

        print('Loss age pretrain {} val MSE {}'.format(avg_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained('./pretrained/age/' + group + '/' + str(fold) + '/.')


def train_classifi(model, train_data, val_data, num_epochs, group, fold):
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=training_args.learning_rate)
    accelerator = Accelerator(fp16=training_args.fp16)
    model, optimizer, train_data = accelerator.prepare(model, optimizer, train_data)

    num_training_steps = num_epochs * len(train_data)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=1,
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
                    # batch[1] = batch[1].to(device)
                    # batch[0] = [v.to(device) for v in batch[0]]
                    batch = [v.to(device) for v in batch]
                    outputs = model(batch[0], labels=batch[1])
                    loss = outputs.loss
                    val_loss += loss.item()
                    tepoch.set_postfix(loss=loss.item())
        avg_val_loss = val_loss / len(val_data)

        print('Loss training {} val loss {}'.format(avg_loss, avg_val_loss))

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained('./checkpoints/' + group + '/vit_cla/' + str(fold) + '/.')


if __name__ == '__main__':
    group = 'wholeset/spontaneous'
    for fold in range(1, 6):
        # training_args.per_device_train_batch_size = 1
        # training_args.gradient_accumulation_steps = 1
        #
        # config = VideoMAEConfig(image_size=112)
        # model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base", config=config)
        # label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
        # pretrain_label = list(set([i for trip in label for i in trip]))
        # pretrain_set = Pretrained(data_path, pretrain_label, feature_extractor)
        # pretrain_data = DataLoader(
        #     pretrain_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
        #     pin_memory=True)
        #
        # pretrain(model, pretrain_data, 3, group, fold)
        #
        # training_args.per_device_train_batch_size = 4
        # training_args.gradient_accumulation_steps = 4
        #
        # config = VideoMAEConfig(image_size=112, num_labels=1)
        # model = VideoMAEForVideoClassification.from_pretrained(
        #     'C:/Users/13661/PycharmProjects/ViViTforKin/pretrained/' + group + '/' + str(fold) + '/.',
        #     local_files_only=True, config=config)
        # age_set = Age(data_path, pretrain_label, feature_extractor)
        # age_data = DataLoader(
        #     age_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
        #     pin_memory=True)
        #
        # label = np.load('./protocols/' + group + '/test/' + str(fold) + '.npy')
        # val_label = list(set([i for trip in label for i in trip]))
        # val_set = Age(data_path, val_label, feature_extractor)
        # val_data = DataLoader(
        #     val_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
        #     pin_memory=True)
        #
        # pretrain_age(model, age_data, val_data, 3, group, fold)

        training_args.per_device_train_batch_size = 4
        training_args.gradient_accumulation_steps = 4

        model = VideoMAEForVideoClassification.from_pretrained(
            'C:/Users/13661/PycharmProjects/ViViTforKin/pretrained/' + group + '/' + str(fold) + '/.',
            local_files_only=True)
        state_dict = model.state_dict()
        patch_embeddings = state_dict['videomae.embeddings.patch_embeddings.projection.weight']
        state_dict['videomae.embeddings.patch_embeddings.projection.weight'] = torch.cat((patch_embeddings, patch_embeddings), 1)

        config = VideoMAEConfig(image_size=112, num_labels=2, num_channels=6)
        model = VideoMAEForVideoClassification.from_pretrained(
            'C:/Users/13661/PycharmProjects/ViViTforKin/pretrained/age/' + group + '/' + str(fold) + '/.',
            local_files_only=True, ignore_mismatched_sizes=True, state_dict=state_dict, config=config)

        # config = VideoMAEConfig(image_size=112, num_labels=2)
        # model = VideoMAEForKinshipClassification.from_pretrained(
        #     'C:/Users/13661/PycharmProjects/ViViTforKin/pretrained/age/' + group + '/' + str(fold) + '/.',
        #     local_files_only=True, ignore_mismatched_sizes=True, config=config)

        # config = VideoMAEConfig(image_size=112, num_labels=2)
        # model = VideoMAEForKinshipClassification.from_pretrained(
        #     "MCG-NJU/videomae-base", config=config)
        # for param in model.videomae.parameters():
        #     param.requires_grad = False
        # i = 0
        # for child in model.videomae.children():
        #     for c in child.children():
        #         i += 1
        #         if i >= 11:
        #             for param in c.parameters():
        #                 param.requires_grad = True

        train_label = np.load('./protocols/' + group + '/train/' + str(fold) + '.npy')
        train_set = ClassificationCat(data_path, train_label, feature_extractor)
        train_data = DataLoader(
            train_set, batch_size=training_args.per_device_train_batch_size, shuffle=True, num_workers=4,
            pin_memory=True)

        val_label = np.load('./protocols/' + group + '/val/' + str(fold) + '.npy')
        val_set = ClassificationCat(data_path, val_label, feature_extractor)
        val_data = DataLoader(
            val_set, batch_size=1, shuffle=True, num_workers=4,
            pin_memory=True)

        train_classifi(model, train_data, val_data, 20, group, fold)