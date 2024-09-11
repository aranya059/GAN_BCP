import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn

# Define the UNet model (same as your training code)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(UNet, self).__init__()

        self.encoder1 = ConvBlock(in_channels, 64)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 1024)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = ConvBlock(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = ConvBlock(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(128, 64)

        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(self.maxpool(e2))
        e4 = self.encoder4(self.maxpool(e3))
        e5 = self.encoder5(self.maxpool(e4))

        # Decoder
        d4 = self.upconv4(e5)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        out = self.outconv(d1)
        return out

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet(in_channels=1, out_channels=2)
model.load_state_dict(torch.load('56_unet_model.pth'))
model.to(device)
model.eval()

# Define transformations for the input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def segment_images(input_folder, output_folder):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):  # Adjust file types if needed
            image_path = os.path.join(input_folder, file_name)
            image = Image.open(image_path).convert('L')
            image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

            # Perform segmentation
            with torch.no_grad():
                output = model(image)
                output = torch.argmax(output, dim=1).cpu().squeeze().numpy()  # Get the predicted mask

            # Save the result
            result_image = Image.fromarray((output * 255).astype(np.uint8))  # Scale to [0, 255] and convert to image
            result_image.save(os.path.join(output_folder, file_name))

    print("Segmentation completed and results saved.")

# Example usage
input_folder = './data/for_figure/fives_quantitative/original_images'  # Replace with your input folder path
output_folder = './data/for_figure/fives_quantitative/unet_gt_10'  # Replace with your output folder path
segment_images(input_folder, output_folder)
