import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=64):
        super(UNet1D, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, base_filters)
        self.enc2 = self.conv_block(base_filters, base_filters * 2)
        self.enc3 = self.conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self.conv_block(base_filters * 4, base_filters * 8)

        # Bottleneck
        self.bottleneck = self.conv_block(base_filters * 8, base_filters * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose1d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(base_filters * 16, base_filters * 8)

        self.upconv3 = nn.ConvTranspose1d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(base_filters * 8, base_filters * 4)

        self.upconv2 = nn.ConvTranspose1d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(base_filters * 4, base_filters * 2)

        self.upconv1 = nn.ConvTranspose1d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(base_filters * 2, base_filters)

        # Output layer
        self.out_conv = nn.Conv1d(base_filters, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        # print("x shape: ", x.shape)
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool1d(enc1, 2))
        enc3 = self.enc3(F.max_pool1d(enc2, 2))
        enc4 = self.enc4(F.max_pool1d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool1d(enc4, 2))

        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        # Output
        out = self.out_conv(dec1)
        # print("out shape: ", out.shape)
        out_flat = out.view(out.size(0), -1)
        # print("out_flat shape: ", out_flat.shape)

        return out_flat
