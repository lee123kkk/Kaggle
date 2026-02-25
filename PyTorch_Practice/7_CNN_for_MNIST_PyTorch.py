# 7_CNN_for_MNIST_PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 재현성을 위한 시드 고정
random.seed(777)
torch.manual_seed(777)
if device.type == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
training_epochs = 12
batch_size = 128

# 데이터 준비 (ToTensor()가 자동으로 0~1 정규화를 수행하며, 채널 차원을 앞으로 옮겨줍니다)
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

# 모델 정의 (CNN)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        # L1: Input shape (1, 28, 28) -> Conv -> (16, 26, 26) -> Pool -> (16, 13, 13)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # L2: Input shape (16, 13, 13) -> Conv -> (32, 11, 11) -> Pool -> (32, 5, 5)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # L3: Fully Connected
        # Flatten 후 크기: 32(채널) * 5(가로) * 5(세로) = 800
        self.fc = nn.Linear(32 * 5 * 5, 10)
        
        # Keras의 'glorot_normal' (Xavier Normal) 초기화 적용
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1) # Keras의 Flatten() 역할
        out = self.fc(out)
        return out

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
model.train()
for epoch in range(training_epochs):
    for X, Y in data_loader:
        # 이미지를 Flatten하지 않고 2D 형태(Batch, Channel, Height, Width) 그대로 GPU로 전송
        X = X.to(device) 
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        loss = criterion(hypothesis, Y)
        loss.backward()
        optimizer.step()

# 평가 및 무작위 데이터 예측
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0
    
    for X, Y in test_loader:
        X = X.to(device)
        Y = Y.to(device)
        
        outputs = model(X)
        loss = criterion(outputs, Y)
        total_loss += loss.item() * X.size(0)
        
        predicted = torch.argmax(outputs, dim=1)
        total += Y.size(0)
        correct += (predicted == Y).sum().item()
        
    avg_loss = total_loss / total
    accuracy = correct / total
    
    # 10개의 무작위 데이터 예측
    for _ in range(10):
        r = random.randint(0, len(mnist_test) - 1)
        # 단일 이미지를 예측할 때 (1, 1, 28, 28) 형태로 차원을 맞추어 GPU로 전송
        X_single_data = mnist_test.data[r].view(1, 1, 28, 28).float().to(device) / 255.0
        Y_single_data = mnist_test.targets[r].item()
        
        single_prediction = model(X_single_data)
        pred_val = torch.argmax(single_prediction, dim=1).item()
        
        print(f"index: {r} actual y: {Y_single_data} predicted y: {pred_val}")
        
    print('loss: ', avg_loss)
    print('accuracy: ', accuracy)

#=========================================================
# CNN pytorch변환

# Keras는 이미지 데이터를 (높이, 너비, 채널)형태로 다루지만 
# PyTorch는 (채널, 높이 너비)형태를 요구한다.

# Keras는 Flatten()레이어를 넣으면 다음 Dense 레이어의 입력 크기를 자동으로 계산하지만, 
# pytorch는 개발자가 합성곱과 풀링을 거친 후의 최종 크기를 직접 계산해서 nn.Linear에 넣어야 한다.


