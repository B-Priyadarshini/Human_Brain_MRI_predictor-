import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SegNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SegNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)

        self.unpool = nn.MaxUnpool2d(2, 2)
        self.dec3 = self.conv_block(256, 128)
        self.dec2 = self.conv_block(128, 64)
        self.dec1 = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x1p, idx1 = self.pool(x1)

        x2 = self.enc2(x1p)
        x2p, idx2 = self.pool(x2)

        x3 = self.enc3(x2p)
        x3p, idx3 = self.pool(x3)

        x3u = self.unpool(x3p, idx3, output_size=x3.size())
        x3d = self.dec3(x3u)

        x2u = self.unpool(x3d, idx2, output_size=x2.size())
        x2d = self.dec2(x2u)

        x1u = self.unpool(x2d, idx1, output_size=x1.size())
        out = self.dec1(x1u)

        return torch.sigmoid(out)
class BloodCellDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        mask = (mask > 0).float()  # Convert to binary mask
        return image, mask

def train_segnet(image_dir, mask_dir, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = BloodCellDataset(image_dir, mask_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = SegNet().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    # Save the model
    torch.save(model.state_dict(), "segnet_bloodcell.pth")
    print("✅ Training complete! Model saved as segnet_bloodcell.pth")

if __name__ == "__main__":
    image_dir = input("Enter full path to training images: ")
    mask_dir = input("Enter full path to training masks: ")
    train_segnet(image_dir, mask_dir)
