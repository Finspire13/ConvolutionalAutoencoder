# Homework 1 for PKU DL class
# Daochang Liu, 20180402
# My work includs the following:
# 1. Convolutional autoencoder for feature extraction
# 2. Unsupervised learning using T-SNE and hierarchical clustering
#
# Autoencoder trained on training set of MNIST (60K images)
# Clustering on features of testing set (10K images)
# It takes about 10 mins to run
# Only clustering at the last epoch can save time
# Final accuracy: ~93%


import torch.nn as nn

# Convolutional Autoencoder
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()

        # Bottom
        self.bottom_block_1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2))    # 14

        self.bottom_block_2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))    # 7

        self.bottom_block_3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=7),
            nn.BatchNorm2d(32),
            nn.ReLU())          # 1

        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU())          # 1

        # Top
        self.top_block_1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU())          # 7

        self.top_block_2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU())    # 14

        self.top_block_3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid())    # 28

    def forward(self, x):

        out = self.bottom_block_1(x)
        out = self.bottom_block_2(out)
        out = self.bottom_block_3(out)
        
        out = self.middle(out)
        img_code = out.view(out.size(0), -1)

        out = self.top_block_1(out)
        out = self.top_block_2(out)
        out = self.top_block_3(out)

        return out, img_code
