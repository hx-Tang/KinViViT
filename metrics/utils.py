from datetime import datetime

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from .verification import evaluate

plt.switch_backend('agg')
import numpy as np
from PIL import Image
import io
import sklearn
import time


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf


def test_forward(device, backbone, data_set):
    backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode
    # embed()
    # last_time1 = time.time()
    forward_time = 0
    carray = data_set[0]
    # print("carray:",carray.shape)
    idx = 0
    with torch.no_grad():
        while idx < 2000:
            batch = carray[idx:idx + 1]
            batch_device = batch.to(device)
            last_time = time.time()
            backbone(batch_device)
            forward_time += time.time() - last_time
            # if idx % 1000 ==0:
            #    print(idx, forward_time)
            idx += 1
    print("forward_time", 2000, forward_time, 2000 / forward_time)
    return forward_time


def perform_val(multi_gpu, device, embedding_size, batch_size, backbone, data_set, issame, nrof_folds=10):
    if multi_gpu:
        backbone = backbone.module  # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                # last_time = time.time()
                embeddings[idx:idx + batch_size] = backbone(batch.to(device)).cpu()
                # batch_time = time.time() - last_time
                # print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor


def perform_val_deit(multi_gpu, device, embedding_size, batch_size, backbone, dis_token, data_set, issame,
                     nrof_folds=10):
    if multi_gpu:
        backbone = backbone.module  # unpackage model from DataParallel
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)
    backbone.eval()  # switch to evaluation mode

    embeddings_list = []
    for carray in data_set:
        idx = 0
        embeddings = np.zeros([len(carray), embedding_size])
        with torch.no_grad():
            while idx + batch_size <= len(carray):
                batch = carray[idx:idx + batch_size]
                # last_time = time.time()
                # embed()
                fea, token = backbone(batch.to(device), dis_token.to(device))
                embeddings[idx:idx + batch_size] = fea.cpu()
                # batch_time = time.time() - last_time
                # print("batch_time", batch_size, batch_time)
                idx += batch_size
            if idx < len(carray):
                batch = carray[idx:]
                embeddings[idx:] = backbone(batch.to(device)).cpu()
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)

    tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    roc_curve_tensor = transforms.ToTensor()(roc_curve)

    return accuracy.mean(), accuracy.std(), _xnorm, best_thresholds.mean(), roc_curve_tensor


def buffer_val(writer, db_name, acc, std, xnorm, best_threshold, roc_curve_tensor, batch):
    writer.add_scalar('Accuracy/{}_Accuracy'.format(db_name), acc, batch)
    writer.add_scalar('Std/{}_Std'.format(db_name), std, batch)
    writer.add_scalar('XNorm/{}_XNorm'.format(db_name), xnorm, batch)
    writer.add_scalar('Threshold/{}_Best_Threshold'.format(db_name), best_threshold, batch)
    writer.add_image('ROC/{}_ROC_Curve'.format(db_name), roc_curve_tensor, batch)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
