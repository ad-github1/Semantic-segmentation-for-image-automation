import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import cv2, os
import numpy as np
from model.enet import ENet
from tqdm import tqdm

class CityscapesSample(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace('_leftImg8bit', '_gtFine_labelIds'))
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            image = self.transform(image)

        return image, torch.from_numpy(mask).long()

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CityscapesSample('data/sample_cityscapes/images', 'data/sample_cityscapes/masks', transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ENet(num_classes=20).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    model.train()
    epoch_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), 'enet_cityscapes.pth')
print(" Training complete! Model saved.")
