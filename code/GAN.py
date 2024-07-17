import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 144 * 88 * 2),  # Adjusted output size for two horizontal images (image + label)
            nn.Tanh()
        )

    def forward(self, z):
        output = self.model(z)
        return output.view(-1, 2, 144, 88)  # Reshape to [batch_size, 2 channels, width, height] for horizontal format

# Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(144 * 88 * 2, 512),  # Input size for horizontal images
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Custom Dataset Class to Load Images and Labels
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_list = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if 'image' in file:
                    img_path = os.path.join(subdir, file)
                    label_path = os.path.join(subdir, file.replace('image', 'label'))
                    self.img_list.append((img_path, label_path))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path, label_path = self.img_list[idx]
        image = Image.open(img_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return image, label

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Transformations
transform = transforms.Compose([
    transforms.Resize((144, 88)),  # Resize to horizontal format
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Setup Dataset and DataLoader
dataset = CustomDataset('./Extracted images/LA_Extracted_Images_Labels', transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
adversarial_loss = nn.BCELoss()

# Training Loop
num_epochs = 500
save_interval = 100  # Save every 100 epochs
for epoch in range(num_epochs):
    for images, labels in dataloader:
        real_imgs = torch.cat([images, labels], dim=1).to(device)
        batch_size = real_imgs.size(0)
        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Generator
        z = torch.randn(batch_size, 100, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Discriminator Loss: {d_loss.item()}, Generator Loss: {g_loss.item()}')

    # Generate and Save Synthetic Images and Labels every 100 epochs
    if (epoch + 1) % save_interval == 0:
        output_path = f'./generated_images_and_labels_epoch_{epoch + 1}'
        os.makedirs(output_path, exist_ok=True)
        with torch.no_grad():
            z = torch.randn(250, 100, device=device)  # Generate 250 images
            generated_data = generator(z)
            for i in range(250):
                image = generated_data[i][0].cpu().numpy()
                label = generated_data[i][1].cpu().numpy()
                plt.imsave(f'{output_path}/generated_image_{i}.png', image, cmap='gray')
                plt.imsave(f'{output_path}/generated_label_{i}.png', label, cmap='gray')

print("Generated images and labels are saved in separate epoch folders.")
