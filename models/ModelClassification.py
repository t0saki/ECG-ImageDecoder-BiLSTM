import torch.nn as nn
# from models.EfficientNetV2_BiLSTM_1D import EfficientNetV2_BiLSTM_1D
from models.MobileNetV3_BiLSTM_1D import MobileNetV3_BiLSTM_1D as model


class ModelClassification(model):
    def __init__(self, num_classes=1000, output_classes=20, hidden_size=256):
        super(ModelClassification, self).__init__(
            num_classes=num_classes, output_size=output_classes, hidden_size=hidden_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = super().forward(x)
        x = self.softmax(x)
        return x
