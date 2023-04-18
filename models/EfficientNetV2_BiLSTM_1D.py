import torch.nn as nn
import torchvision.models as models


class EfficientNetV2_BiLSTM_1D(nn.Module):
    def __init__(self, num_classes=1000, output_size=2, hidden_size=256):
        super(EfficientNetV2_BiLSTM_1D, self).__init__()

        # Load the pre-trained EfficientNetV2-S model
        self.efficientnetv2 = models.efficientnet_v2_s(
            weights='DEFAULT', num_classes=num_classes)

        # Replace the 2D convolutional layers with 1D convolutional layers
        self.conv_stem = nn.Conv1d(
            in_channels=1, out_channels=24, kernel_size=3, stride=2, padding=1, bias=False)
            

        # Remove the classifier layer

        # Add a BiLSTM layer
        self.bilstm = nn.LSTM(input_size=1408, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)

        # Add 2 fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size//2)
        self.fc2 = nn.Linear(hidden_size//2, output_size)

    def forward(self, x):
        # Treat each time step in the 1D ECG signal as a separate channel
        x = x.unsqueeze(1)

        x = self.conv_stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.efficientnetv2._avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.efficientnetv2._dropout(x)
        x = self.efficientnetv2._fc(x)
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze()
