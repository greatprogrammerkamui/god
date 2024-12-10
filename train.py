import time
import matplotlib.pyplot as plt
import torch
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import models

ds_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32,scale=True)
])

ds_train = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ds_transform
)
ds_test = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ds_transform
)
batch_size = 64
dataloader_train = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    shuffle=True
)

dataloader_test = torch.utils.data.DataLoader(
    ds_test,
    batch_size=batch_size
)
for image_batch, label_batch in dataloader_test:
    print(image_batch.shape)
    print(label_batch.shape)
    break
model = models.MyModel()

acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.3f}%')

model = models.MyModel()

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-3  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# モデルをインスタンス化する
model = models.MyModel()

# 損失関数（誤差関数・ロス関数）の選択
loss_fn = torch.nn.CrossEntropyLoss()

# 最適化の方法の選択
learning_rate = 1e-3  # 学習率
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 精度を計算
acc_test = models.test_accuracy(model, dataloader_test)
print(f'test accuracy: {acc_test*100:.2f}%')

# 学習回数
n_epochs = 5

# 学習
for k in range(n_epochs):
    print(f'epoch {k+1}/{n_epochs}')

    # 1 epoch の学習
    time_start = time.time()
    loss_train = models.train(model, dataloader_train, loss_fn, optimizer)
    time_end = time.time()
    print(f'train loss: {loss_train}')

    loss_test = models.test(model, dataloader_test, loss_fn)
    print(f'test loss: {loss_test}')

    # 精度を計算する
    acc_train = models.test_accuracy(model, dataloader_train)
    print(f'train accuracy: {acc_train*100:.3f}%')
    acc_test = models.test_accuracy(model, dataloader_test)
    print(f'test accuracy: {acc_test*100:.3f}%')


import matplotlib.pyplot as plt

# モデルをインスタンス化
model = models.MyModel()

# 損失関数とオプティマイザの設定
loss_fn = torch.nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 記録用リスト
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# エポック数
n_epochs = 5

# トレーニングとテスト
for epoch in range(n_epochs):
    print(f'Epoch {epoch + 1}/{n_epochs}')

    # トレーニング
    train_loss = models.train(model, dataloader_train, loss_fn, optimizer)
    train_acc = models.test_accuracy(model, dataloader_train)

    # テスト
    test_loss = models.test(model, dataloader_test, loss_fn)
    test_acc = models.test_accuracy(model, dataloader_test)

    # 結果を記録
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # 結果を出力
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc*100:.2f}%')

# グラフで表示
epochs = range(1, n_epochs + 1)

plt.figure()
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')
plt.show()

plt.figure()
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy Over Epochs')
plt.show()
