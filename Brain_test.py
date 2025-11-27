import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SegNet().to(device)
model.load_state_dict(torch.load("segnet_braincell.pth", map_location=device))
model.eval()

image_path = input("Enter path to your test Brain MRI image: ")

img_color = cv2.imread(image_path)
if img_color is None:
    raise FileNotFoundError(f"❌ Could not load image at {image_path}")

img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_color, (256, 256))
img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY) / 255.0  # Normalize grayscale

img_tensor = torch.tensor(img_gray, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    pred_mask = output.squeeze().cpu().numpy()
    binary_mask = pred_mask > 0.5  # Threshold to get binary mask


masked_image = img_resized.copy()
masked_image[~binary_mask] = 0  # Mask all 3 channels by broadcasting


plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_resized)
plt.title("Original Image (Color)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_mask, cmap='gray')
plt.title("Predicted Mask (Gray)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(masked_image)
plt.title("Segmented Output (Dark Spots)")
plt.axis('off')

plt.tight_layout()
plt.show()
