import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import random
import umap
import record_keeper
from cycler import cycler
from torchvision import datasets, transforms
from datasets.Uva import UVAts, load_label

import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

# Set trunk model and replace the softmax layer with an identity function
trunk = torchvision.models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
trunk = torch.nn.DataParallel(trunk.to(device))

# Set embedder model. This takes in the output of the trunk and outputs 128 dimensional embeddings
embedder = torch.nn.DataParallel(MLP([trunk_output_size, 128]).to(device))

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00004, weight_decay=0.00005)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=0.00004, weight_decay=0.00005
)

# Set the image transforms
train_transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

label_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_Kinship_Labels.txt'
data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

train_label, val_label = load_label(label_path, 0.8, True)

train_dataset = UVAts(data_path, train_label,train_transform, train_transform)
val_dataset = UVAts(data_path, val_label,val_transform, val_transform)



# Set the loss function
loss = losses.TripletMarginLoss(margin=0.2)

# Set the mining function
miner = miners.TripletMarginMiner(margin=0.2)

# # Set the dataloader sampler
# sampler = samplers.MPerClassSampler(
#     original_train.classes, m=1, length_before_new_iter=len(train_dataset)
# )

# Set other training parameters
batch_size = 12
num_epochs = 4

# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    "embedder_optimizer": embedder_optimizer,
}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

record_keeper, _, _ = logging_presets.get_record_keeper(
    "example_logs", "example_tensorboard"
)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"


def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    half = int(umap_embeddings.shape[0] / 2)
    anchors = umap_embeddings[:half]
    posneg = umap_embeddings[half:]
    labels = labels[:half]
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(posneg[idx, 0], posneg[idx, 1], "s", markersize=1)
        plt.plot(anchors[idx, 0], anchors[idx, 1], ".", markersize=1)
    plt.show()


# Create the tester
tester = testers.GlobalTwoStreamEmbeddingSpaceTester(
    end_of_testing_hook=hooks.end_of_testing_hook,
    visualizer=umap.UMAP(n_neighbors=50),
    visualizer_hook=visualizer_hook,
    dataloader_num_workers=2,
    accuracy_calculator=AccuracyCalculator(k="max_bin_count"),
)

end_of_epoch_hook = hooks.end_of_epoch_hook(tester, dataset_dict, model_folder)



trainer = trainers.TwoStreamMetricLoss(
    models,
    optimizers,
    batch_size,
    loss_funcs,
    mining_funcs,
    train_dataset,
    dataloader_num_workers=2,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)

if __name__ == '__main__':
    trainer.train(num_epochs=num_epochs)