import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import os
from PIL import Image

# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义模型
class LargeKernelBlock(nn.Module):
    def __init__(self, filters, dilation_rate=1):
        super(LargeKernelBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=9, padding=4*dilation_rate, dilation=dilation_rate, groups=3)
        self.conv = nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=5, padding=2)
        self.batch_norm = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.depthwise_conv(x))
        x = self.activation(self.conv(x))
        x = self.batch_norm(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, filters, dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=filters, kernel_size=3, padding=1, dilation=dilation_rate)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(filters)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.batch_norm(x)
        x = self.activation(self.conv2(x))
        x = self.batch_norm(x)
        return x

class MyLargeKernel(nn.Module):
    def __init__(self, filters, dilation_rate=1):
        super(MyLargeKernel, self).__init__()
        self.large_kernel1 = LargeKernelBlock(filters, dilation_rate)
        self.conv_block1 = ConvBlock(filters, dilation_rate=1)
        self.pool = nn.MaxPool2d(2)
        
        self.large_kernel2 = LargeKernelBlock(2*filters, dilation_rate // 2)
        self.conv_block2 = ConvBlock(2*filters, dilation_rate=1)
        
        self.large_kernel3 = LargeKernelBlock(4*filters, dilation_rate // 4)
        self.conv_block3 = ConvBlock(4*filters, dilation_rate=1)
        
        self.large_kernel4 = LargeKernelBlock(8*filters, dilation_rate // 8)
        self.conv_block4 = ConvBlock(8*filters, dilation_rate=1)
        
        self.final_conv = nn.Conv2d(8*filters, filters, kernel_size=1, padding=0)
        
    def forward(self, x):
        x1 = self.pool(self.large_kernel1(x))
        y1 = self.pool(self.conv_block1(x))
        inter1_1 = torch.cat((x1, y1), dim=1)
        
        x2 = self.pool(self.large_kernel2(x1))
        y2 = self.pool(self.conv_block2(inter1_1))
        inter2_1 = torch.cat((x2, y2), dim=1)
        
        x3 = self.pool(self.large_kernel3(x2))
        y3 = self.pool(self.conv_block3(inter2_1))
        inter3_1 = torch.cat((x3, y3), dim=1)
        
        x4 = self.pool(self.large_kernel4(x3))
        y4 = self.pool(self.conv_block4(inter3_1))
        inter4_1 = torch.cat((x4, y4), dim=1)
        
        out = self.final_conv(inter4_1)
        return out

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.large_kernel = MyLargeKernel(filters=64, dilation_rate=256)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        x = self.large_kernel(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        x = self.activation(x)
        return x

model = MyModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# 数据集和数据加载器
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

transform = transforms.Compose([
    transforms.Resize((256*9, 256*9)),
    transforms.ToTensor()
])

image_dir = r"E:\Qiyao_Xu\QyXu\test_data\test_array"
mask_dir = r"E:\Qiyao_Xu\QyXu\test_data\test_masks"
batch_size = 10

dataset = CustomDataset(image_dir, mask_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 训练
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("Training finished.")
