import torch.nn as nn
# from models.EfficientNetV2_BiLSTM_1D import EfficientNetV2_BiLSTM_1D
from models.MobileNetV3_BiLSTM_1D import MobileNetV3_BiLSTM_1D as model


class ModelPrediction(model):
    def __init__(self, num_classes=1000, output_size=10, hidden_size=256):
        super(ModelPrediction, self).__init__(
            num_classes=num_classes, output_size=output_size, hidden_size=hidden_size)

    def forward(self, x):
        x = super().forward(x)
        return x
