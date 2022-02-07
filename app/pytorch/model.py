from numpy import number
from torch import nn, flatten


class Model(nn.Module):
    def __init__(self, base_model=None, number_of_classes=26):
        super(Model, self).__init__()

        self.base_model = base_model
        self.number_of_classes = number_of_classes
        self.regressor = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, self.number_of_classes),
        )

        self.base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        bboxes = self.regressor(features)
        class_logits = self.classifier(features)
        return bboxes, class_logits
