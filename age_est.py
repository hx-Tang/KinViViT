import torch
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
from tqdm import tqdm

from models.mobilenetv2 import MobileNetV2
from datasets.Uva import UVAage, load_age
import torchvision.transforms as transforms


def main():
    device = torch.device('cuda')

    label_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/UvA-NEMO_Smile_Database_File_Details.txt'
    data_path = 'D:/文档/硕士/Thesis/UvA-NEMO_SMILE_DATABASE/aligned'

    label = load_age(label_path)
    random.shuffle(label)

    transform = transforms.Compose(
        [transforms.ToTensor()])

    trainset = UVAage(data_path, label[:int(0.8*len(label))], transform)
    valset = UVAage(data_path, label[int(0.8*len(label)):], transform)

    train_loader = DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        valset, batch_size=16, shuffle=True, num_workers=4)

    model = MobileNetV2(num_classes=100, sample_size=112, width_mult=1.)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adadelta(params, lr=1.0)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    loss_fn = torch.nn.CrossEntropyLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    EPOCHS = 10

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


def train_one_epoch(epoch_index, tb_writer, model, train_loader, optimizer, loss_fn, device):
    running_loss = 0.

    model.train()
    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    with tqdm(train_loader, unit="batch") as tepoch:
        for data in tepoch:
            # Every data instance is an input + label pair
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Make predictions for this batch
            outputs = model(inputs)

            # Compute the loss and its gradients
            loss = loss_fn(outputs, labels)
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
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                running_vloss += loss

                vepoch.set_postfix(loss=loss.item())

    avg_vloss = running_vloss / len(val_loader)

    return avg_vloss


if __name__ == "__main__":
    main()