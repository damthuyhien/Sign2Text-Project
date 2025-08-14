import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import SignSequenceNet

# transform ảnh
transform = transforms.Compose([
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

# dataset 29 class
train_dataset = datasets.ImageFolder('data/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SignSequenceNet(num_classes=29).to(device)

# loss + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train
for epoch in range(5):
    print(f"Starting epoch {epoch+1}")
    for i, (images, labels) in enumerate(train_loader):
        print(f" Batch {i+1}, images.shape = {images.shape}, labels = {labels}")
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")


# lưu model
torch.save(model.state_dict(), 'saved_model.pth')
print("Model saved as saved_model.pth")
