from .model import Model
from .dataset import Dataset

import torch, cv2, PIL
import torch.nn as nn
from torch import optim
from torchvision.models import resnet50
import numpy as np
from torch.utils.data import DataLoader

from matplotlib import pyplot as plt


def collate(batch):
    images = []
    bboxes = []
    labels = []
    for img, box, label in batch:
        images.append([img])
        bboxes.append([box])
        labels.append([label])

    images = np.concatenate(images, axis=0)
    images = torch.from_numpy(images.transpose(0, 3, 1, 2)).div(255.0)

    bboxes = np.concatenate(bboxes, axis=0)
    bboxes = torch.from_numpy(bboxes)

    labels = np.concatenate(labels, axis=0)
    labels = torch.from_numpy(labels)

    return images, bboxes, labels


def train(cnn_model, device, dataset, epochs=5, batch_size=10):
    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8, collate_fn=collate
    )

    global_step = 0

    optimizer = optim.Adam(
        cnn_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08
    )

    labels_loss_function = nn.CrossEntropyLoss()
    bboxes_loss_function = nn.MSELoss()

    cnn_model = cnn_model.double()

    total_training_loss = 0
    total_correct_predictions = 0

    for epoch in range(epochs):
        epoch_step = 0
        for i, (images, bboxes, labels) in enumerate(train_loader):
            global_step += 1
            epoch_step += 1

            images = images.double()
            (images, labels, bboxes) = (
                images.to(device),
                labels.to(device),
                bboxes.to(device),
            )

            bboxes_pred, labels_pred = cnn_model(images)

            bboxes_loss = bboxes_loss_function(bboxes_pred.unsqueeze(1), bboxes)
            labels_loss = labels_loss_function(labels_pred, labels)
            total_loss = bboxes_loss + labels_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_training_loss += total_loss

            correct_labels = np.argmax(labels.detach().numpy(), axis=1)
            predicted_labels = np.argmax(labels_pred.detach().numpy(), axis=1)
            number_of_correct_predictions = sum(
                [i == j for i, j in zip(correct_labels, predicted_labels)]
            )
            total_correct_predictions += number_of_correct_predictions

            last_training_accuracy = number_of_correct_predictions / batch_size * 100

            total_training_accuracy = (
                total_correct_predictions / (batch_size * global_step) * 100
            )

            print(
                f"""
            epochs: {epoch}
            number of steps: {global_step}
            total loss: {total_loss}
            total training loss: {total_training_loss}
            total accuracy: {total_training_accuracy}%
            last accuracy: {last_training_accuracy}%
            """
            )

    torch.save(cnn_model.state_dict(), f"models/model.pth")


if __name__ == "__main__":

    resnet = resnet50(pretrained=True)

    for param in resnet.parameters():
        param.requires_grad = False

    cnn_model = Model(base_model=resnet, number_of_classes=26)

    dataset = Dataset(
        label_path="data/american_signlanguage/train/_annotations.txt",
        train_data_path="data/american_signlanguage/train",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn_model.to(device=device)

    train(cnn_model=cnn_model, device=device, dataset=dataset, epochs=5, batch_size=32)
