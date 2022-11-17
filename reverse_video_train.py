import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import kornia
import torchinfo

from video_model import ConvLSTM, VideoModel
from utils import Checkpoint


def ssim_loss(output, target):
    losses = torch.zeros(output.shape[0])
    for idx in range(output.shape[0]):
        losses[idx] = kornia.losses.ssim_loss(output[idx], target[idx], 5)
    return losses.mean()


def train(model, dataloader, optimizer, device):
    train_loss = []

    model.train()
    pbar = tqdm(dataloader)
    for data in pbar:
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        output = model(inputs)

        loss = ssim_loss(output, labels)
        loss.backward()
        optimizer.step()

        pbar.set_description(f"Loss {round(loss.item(), 4)}")
        train_loss.append(loss.item())
    return np.mean(train_loss)


def test(model, dataloader, optimizer, device):
    test_loss = []

    model.eval()
    with torch.no_grad():
        pbar = tqdm(dataloader)
        for data in pbar:
            inputs, labels = data[0].to(device), data[1].to(device)

            output = model(inputs)

            loss = ssim_loss(output, labels)

            pbar.set_description(f"Loss {round(loss.item(), 4)}")
            test_loss.append(loss.item())
    return np.mean(test_loss)


def create_labels(features, input_width, prediction_length):
    total_window_size = input_width + prediction_length
    inputs = features[:, :input_width, :, :, :]
    labels = features[:, input_width:total_window_size, :, :, :]

    return inputs, labels


class VideoDataset(Dataset):
    def __init__(self, fname, seq_len, pred_len):
        self.data = np.load(fname, allow_pickle=True)
        self.data = np.flip(self.data, axis=1)
        self.data = self.data / 255
        self.inputs, self.labels = create_labels(self.data, seq_len, pred_len)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):  # Return an item
        data = self.inputs[idx]
        label = self.labels[idx]

        return data.astype(np.float32), label.astype(np.float32)


def main(seq_len, pred_len, lr, batch_size, filters, export_name):
    device = torch.device("cuda")
    epochs = 10

    # input_chan, hidden_chan, kernel_size, batch_first, bias, return_all_layers
    convlstm = ConvLSTM(1, filters, (5, 5), 3, True, True, False)
    vid_model = VideoModel(convlstm, (3, 3), seq_len, pred_len)
    # torchinfo.summary(vid_model, (batch_size, seq_len, 1, 96, 160))

    vid_model.to(device)
    optimizer = optim.Adam(vid_model.parameters(), lr=lr)

    train_dataset = VideoDataset("video_train_torch.npy", seq_len, pred_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = VideoDataset("video_val_torch.npy", seq_len, pred_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    checkpoint = Checkpoint("torch_models/", "Loss", mode="min", initial_best=1)
    _train_loss, _val_loss = [], []

    for e in range(epochs):
        print(f"##### Epoch {e + 1}/{epochs} #####")
        train_loss = train(vid_model, train_loader, optimizer, device)
        _train_loss.append(train_loss)
        print("Training loss: %.5f" % (train_loss))
        val_loss = test(vid_model, val_loader, optimizer, device)
        _val_loss.append(val_loss)
        print("Validation loss: %.5f" % (val_loss))
        checkpoint.on_epoch_end(val_loss, model=("reverse_convlstm", export_name, vid_model))
        if val_loss < 0.075:
            print(f"Loss is super low. Overfitting?")
            break

    with open(f"torch_models/reverse_{export_name}.txt", "w") as f:
        for (t, v) in zip(_train_loss, _val_loss):
            f.write(f"Train: {round(t, 5)}, Val: {round(v, 5)}\n")


if __name__ == "__main__":
    # history, pred, lr, bs, hidden_dim, export_name
    runs = [[15, 5, 0.001, 8, [16, 18, 20], "model"]
            ]

    for run in runs:
        print(f"Training {run[-1]}...")
        print(run)
        os.system(f"python preprocessing.py {run[0]} {run[1]} {run[0]}")
        main(*run)
