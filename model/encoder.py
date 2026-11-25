import torch.nn as nn
import torch.nn.functional as F


from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import EMB, ENCODER_INPUT_SIZE, ENCODER_CHANNELS, LEAKY_SLOPE



class Encoder(nn.Module): #small cnn that turns an image into an embedding vector, and also gives the spatial feature map

    def __init__(self, emb=EMB):
        super().__init__()

        layers = [] #a conv and a leaky relu for each pair of neighbouring channel numbers
        for in_channels, out_channels in zip(ENCODER_CHANNELS[:-1], ENCODER_CHANNELS[1:]):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)) #halves the height and width
            layers.append(nn.LeakyReLU(LEAKY_SLOPE, inplace=True))

        self.net = nn.Sequential(*layers) #shape: (B, 3, 256, 256) -> (B, 32, 8, 8) with the current config
        self.fc = nn.Linear(ENCODER_CHANNELS[-1], emb) #shape: (B, 32) -> (B, emb)

    def forward(self, x, return_features=False):
        #x shape: (B, 3, H, W)
        x = F.interpolate(x, size=(ENCODER_INPUT_SIZE, ENCODER_INPUT_SIZE), mode="bilinear", align_corners=False) #resize, shape: (B, 3, 256, 256)

        features = self.net(x) #shape: (B, 32, 8, 8), still has the spatial layout of the image
        pooled = features.mean(dim=(2, 3)) #global average pooling, shape: (B, 32)

        e = self.fc(pooled) #shape: (B, emb)
        e = F.leaky_relu(e, LEAKY_SLOPE) #the embedding

        if return_features:
            return e, features #the gates use the feature map as extra spatial input
        return e