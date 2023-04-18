import torch.nn as nn
import torchvision.models as models


class MobileNetV3_BiLSTM_1D(nn.Module):
    def __init__(self, num_classes=1000, output_size=2, hidden_size=256):
        super(MobileNetV3_BiLSTM_1D, self).__init__()

        # Load the pre-trained MobileNetV3-Small model
        self.mobilenetv3 = models.mobilenet_v3_small(
            pretrained=True, num_classes=num_classes
        )

        # Modify the first convolutional layer to accept 1D input
        self.mobilenetv3.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=(3, 1),
            stride=(2, 1),
            padding=(1, 0),
            bias=False,
        )

        # Remove the final softmax and classifier layers
        self.mobilenetv3.classifier = nn.Identity()

        # Add a BiLSTM layer
        self.bilstm = nn.LSTM(input_size=576, hidden_size=hidden_size,
                              num_layers=1, batch_first=True, bidirectional=True)

        # Add 2 fully connected layers
        self.fc1 = nn.Linear(hidden_size*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Treat each time step in the 1D ECG signal as a separate channel
        x = x.unsqueeze(1)
        x = self.mobilenetv3(x)
        x = x.view(x.size(0), -1)
        x = x.unsqueeze(-1).permute(0, 2, 1)
        x, _ = self.bilstm(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze()
