# 6_NN_for_MNIST_PyTorch

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
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

learning_rate = 0.001
batch_size = 100
training_epochs = 15
nb_classes = 10

# 1. DataLoader를 이용한 데이터 세팅 (Batch 처리)
# PyTorch는 torchvision에서 데이터를 받고, DataLoader를 통해 배치 단위로 나눕니다.
mnist_train = dsets.MNIST(root='MNIST_data/', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = dsets.MNIST(root='MNIST_data/', train=False, transform=transforms.ToTensor(), download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=mnist_test, batch_size=batch_size, shuffle=False)

# 2. 모델 정의 (ReLU 활성화 함수 추가)
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),            # 첫 번째 은닉층 활성화
    nn.Linear(256, 256),
    nn.ReLU(),            # 두 번째 은닉층 활성화
    nn.Linear(256, nb_classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3. 모델 학습 (배치 단위 반복 추가)
model.train()
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    # 미니 배치 단위로 꺼내옵니다 (X: 이미지, Y: 라벨)
    for X, Y in data_loader:
        # 이미지를 1차원(784)으로 평탄화(Flatten)하여 GPU로 전송
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

# 4. 평가 및 무작위 데이터 예측
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    total_loss = 0
    
    # Keras의 evaluate() 역할
    for X, Y in test_loader:
        X = X.view(-1, 28 * 28).to(device)
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
        # PyTorch 테스트 데이터셋의 원본을 가져올 때도 형태 변환과 정규화(/255.0)가 필요합니다.
        X_single_data = mnist_test.data[r].view(-1, 28 * 28).float().to(device) / 255.0
        Y_single_data = mnist_test.targets[r].item()
        
        single_prediction = model(X_single_data)
        pred_val = torch.argmax(single_prediction, dim=1).item()
        
        print(f"index: {r} actual y: {Y_single_data} predicted y: {pred_val}")
        
    print('loss: ', avg_loss)
    print('accuracy: ', accuracy)
#===========================================================
# MNIST pytorch로 변환
    
# Keras에서는 fit()의 파라미터로 batch_size=100만 주면 자동으로 데이터를 쪼개어 학습하지만, 
# Pytorch에서는 DataLoader란느 객체를 생성해 데이터를 미리 100개씩 묶어두고, 
# 학습 루프안에서 for X, Y in data_loader : 의 형태로 직접 꺼내 써야 한다. 

# .view()를 이용한 이미지 평탄화: 
# Keras 코드 상단에서 x_train.reshape()를 통해 데이터를 1차원으로 쭉 폈던 작업을 
# PyTorch에서는 배치 학습 루프 내부에서 X.view(-1, 28 * 28)를 이용해 동적으로 차원을 변환한다.

# 활성화 함수(nn.ReLU)의 명시적 배치: 
# Keras에서는 레이어 파라미터로 activation='relu'를 주입했지만, 
# PyTorch에서는 nn.Sequential 내부에 독립적인 nn.ReLU() 층을 레고 블록 끼우듯 교차해서 넣어주어야 한다.

