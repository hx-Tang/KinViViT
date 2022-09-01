import sklearn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from datasets.Uva import UVAtriplet
from models import video_transformer
from metrics import verification
from metrics.utils import gen_plot
from models import video_transformer, vivit, mobilenetv2

import numpy as np
from tqdm import tqdm
from PIL import Image

def eval():
    device = torch.device('cuda')

    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

    # model = video_transformer.ViViT(num_frames=16,
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
    # pre_train_dict = torch.load('checkpoint/vivit/40fold1.pth')
    # model.load_state_dict(pre_train_dict)

    model = mobilenetv2.MobileNetV2(num_classes=128, sample_size=112, width_mult=1.)
    pre_train_dict = torch.load('checkpoint/bak2/3dcnn/40fold1.pth')
    model.load_state_dict(pre_train_dict)

    model.to(device)

    test_label = np.load('folds/test_label_fold1.npy')

    transform = transforms.Compose(
        [transforms.ToTensor()])

    valset = UVAtriplet(data_path, test_label, transform)

    val_loader = DataLoader(
        valset, batch_size=1, shuffle=True, num_workers=1)

    embeddings1, embeddings2, is_same = run(model, val_loader, device)
    embeddings1 = sklearn.preprocessing.normalize(embeddings1)
    embeddings2 = sklearn.preprocessing.normalize(embeddings2)

    tpr, fpr, accuracy, best_thresholds = verification.evaluate(embeddings1, embeddings2, is_same)

    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)

    print(accuracy.mean(), accuracy.std(), best_thresholds.mean())

    roc_curve.show()


def run(model, val_loader, device):
    model.eval()

    embeddings1 = []
    embeddings2 = []
    is_same = []

    with torch.no_grad():
        with tqdm(val_loader, unit="batch") as vepoch:
            for data in vepoch:
                anchor, pos, neg = data
                anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)

                anchor_out = model(anchor)
                pos_out = model(pos)
                neg_out = model(neg)

                anchor_out = anchor_out.cpu().detach().numpy()
                pos_out = pos_out.cpu().detach().numpy()
                neg_out = neg_out.cpu().detach().numpy()

                embeddings1.append(anchor_out[0])
                embeddings2.append(pos_out[0])
                is_same.append(1)
                embeddings1.append(anchor_out[0])
                embeddings2.append(neg_out[0])
                is_same.append(0)

    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    is_same = np.array(is_same)

    return embeddings1, embeddings2, is_same


if __name__ == '__main__':
    eval()