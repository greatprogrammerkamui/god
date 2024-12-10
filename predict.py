import matplotlib.pyplot as plt

import torch
from torchvision import datasets,transforms
import torchvision.transforms.v2 as transforms

import models

model = models.MyModel()
print(model)

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),  # 画像をTensorに変換
        transforms.Lambda(lambda x: x.to(torch.float32) * 255)  # データ型変換（例: float32にする）
    ])
)

# データの確認
image, target = ds_train[0]
print(f"Image shape: {image.shape}, Target: {target}")

model.eval()
with torch.no_grad():
    logits = model(image)

print(logits)

plt.bar(range(len(logits[0])),logits[0])
plt.show()
probs = logits.softmax(dim=1)
plt.bar(range(len(probs[0])),probs[0])
plt.ylim(0,1)
plt.show()