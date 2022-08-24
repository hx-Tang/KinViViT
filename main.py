import random
from datetime import datetime
import numpy as np

import torch
import torchvision.transforms as transforms
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.Uva import UVAtriplet
from models import video_transformer, vivit, mobilenetv2


def main():
    device = torch.device('cuda')

    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

    file_detail = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_File_Details.txt'
    kin_label = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Kinship_Labels.txt'

    # file_detail = UVAtriplet.read_file(file_detail)
    # kin_label = UVAtriplet.read_file(kin_label)
    #
    # random.shuffle(kin_label)
    #
    # train_label = UVAtriplet.load_label(file_detail, kin_label[:int(0.8 * len(kin_label))])
    # test_label = UVAtriplet.load_label(file_detail, kin_label[int(0.8 * len(kin_label)):], False)
    #
    # random.shuffle(train_label)
    # np.save('train_label.npy', train_label)
    #
    # random.shuffle(test_label)
    # np.save('test_label.npy', test_label)
    train_label = np.load('train_label.npy')
    test_label = np.load('test_label.npy')

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = UVAtriplet(data_path, train_label, transform)
    valset = UVAtriplet(data_path, test_label, transform)

    train_loader = DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        valset, batch_size=4, shuffle=True, num_workers=4)

    model = mobilenetv2.MobileNetV2(num_classes=128, sample_size=112, width_mult=1.)
    pre_train_dict = torch.load('pretrain/kinetics_mobilenetv2_1.0x_RGB_16_best.pth')

    model_dict = {k[7:]: v for k, v in pre_train_dict['state_dict'].items()}

    model_dict.pop('classifier.1.bias')
    model_dict.pop('classifier.1.weight')

    model.load_state_dict(model_dict, strict=False)

    # model = ViViT(112, 16, 128, 32)

    # model = video_transformer.ViViT(num_frames=32,
    #                                 img_size=112,
    #                                 patch_size=16,
    #                                 embed_dims=128,
    #                                 num_heads=16,
    #                                 num_transformer_layers=4,
    #                                 pretrained='pretrain/vivit_model.pth',
    #                                 weights_from='kinetics',
    #                                 attention_type='divided_space_time',
    #                                 use_learnable_pos_emb=False,
    #                                 return_cls_token=False)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adadelta(params, lr=0.1
                         )
    lr_scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    loss_fn = torch.nn.TripletMarginLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/triplet_trainer_{}'.format(timestamp))

    EPOCHS = 60

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch + 1))
        # train for one epoch, printing every 10 iterations
        avg_loss = train_one_epoch(epoch, writer, model, train_loader, optimizer, loss_fn, device)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        avg_vloss = evaluate(model, val_loader, loss_fn, device)

        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch + 1)
        writer.flush()

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), 'checkpoint/' + str(epoch + 1) + '_3d_pretrain_argu.pth')


def train_one_epoch(epoch_index, tb_writer, model, train_loader, optimizer, loss_fn, device):
    running_loss = 0.

    model.train()
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    with tqdm(train_loader, unit="batch") as tepoch:
        for data in tepoch:
            # Every data instance is an input + label pair
            anchor, pos, neg = data
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            anchor_out = model(anchor)
            pos_out = model(pos)
            neg_out = model(neg)

            # Compute the loss and its gradients
            loss = loss_fn(anchor_out, pos_out, neg_out)
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            tepoch.set_postfix(loss=loss.item())

    last_loss = running_loss / len(train_loader) # loss per batch
    tb_x = epoch_index * len(train_loader)
    tb_writer.add_scalar('Loss/train', last_loss, tb_x)

    return last_loss


def evaluate(model, val_loader, loss_fn, device):
    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as vepoch:
            for data in vepoch:
                anchor, pos, neg = data
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                anchor_out = model(anchor)
                pos_out = model(pos)
                neg_out = model(neg)

                loss = loss_fn(anchor_out, pos_out, neg_out)
                running_vloss += loss

                vepoch.set_postfix(loss=loss.item())

    avg_vloss = running_vloss / len(val_loader)

    return avg_vloss


if __name__ == "__main__":
    main()
