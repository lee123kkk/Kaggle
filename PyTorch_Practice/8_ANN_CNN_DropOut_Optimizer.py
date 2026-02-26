#8_ANN_CNN_DropOut_Optimizer

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# [추가됨] 훈련 과정을 기록하고 대시보드로 보여주는 텐서보드 라이브러리
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# 1. 설정 및 데이터 준비
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 텐서보드 설정: 'runs/cifar10_experiment_1' 폴더에 로그 파일이 차곡차곡 쌓입니다.
writer = SummaryWriter('runs/cifar10_experiment_1')

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# [텐서보드 1] 모델이 학습할 이미지 배치를 그리드 형태로 묶어서 텐서보드에 '사진'으로 기록합니다.
dataiter = iter(train_loader)
images, labels = next(dataiter)
img_grid = torchvision.utils.make_grid(images)
writer.add_image('CIFAR-10_Images', img_grid)

# ==========================================
# 2. 성능 향상 기법이 적용된 CNN 모델 설계
# ==========================================
class ConceptCNN(nn.Module):
    def __init__(self):
        super(ConceptCNN, self).__init__()
        
        # Layer 1: Conv -> BatchNorm -> LeakyReLU -> MaxPool
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),       # [성능] 각 배치별로 분포를 정규화하여 학습 속도와 안정성을 대폭 끌어올림
            nn.LeakyReLU(0.1),        # [성능] 0 이하의 값도 살짝 흘려보내어 Dead ReLU(노드가 죽는 현상) 방지
            nn.MaxPool2d(2, 2)
        )

        # Layer 2 & 3: 더 깊은 특징 추출
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.flatten = nn.Flatten()
        
        # Fully Connected Layer (분류기)
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),          # [성능] 노드를 50% 확률로 끄면서 과적합(Overfitting)을 강력하게 방지
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.flatten(x)
        x = self.fc_layer(x)
        return x

model = ConceptCNN().to(device)

# [텐서보드 2] 설계한 모델의 층계도(Graph)를 텐서보드에 기록하여 시각적으로 확인합니다.
writer.add_graph(model, images.to(device))

# ==========================================
# 3. 손실 함수, 옵티마이저, 학습 루프
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0  
    total = 0    
    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # 텐서보드에 기록할 정확도 계산
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # [텐서보드 3] 에포크가 끝날 때마다 평균 Loss와 Accuracy를 점으로 찍어 그래프를 그립니다.
    avg_loss = running_loss / len(train_loader)
    avg_acc = 100 * correct / total
    
    writer.add_scalar('Loss/train', avg_loss, epoch)
    writer.add_scalar('Accuracy/train', avg_acc, epoch)

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f}")

writer.close()

# ==========================================
# 4. 성능 평가
# ==========================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy on CIFAR-10 test images: {100 * correct / total:.2f}%')


# ==========================================
# [보너스] 전이 학습 (Transfer Learning) 예시 코드
# 이미지 우측 하단 '전이 학습' 개념 구현
# ==========================================
"""
# torchvision에서 제공하는 사전 학습된 모델(ResNet18) 불러오기
transfer_model = torchvision.models.resnet18(pretrained=True)

# 전이 학습을 위해 마지막 계층(FC)만 CIFAR-10(10개 클래스)에 맞게 수정
num_ftrs = transfer_model.fc.in_features
transfer_model.fc = nn.Linear(num_ftrs, 10)

transfer_model = transfer_model.to(device)
# 이후 위와 동일하게 학습 진행...
"""


#======================================================================
# BatchNorm2d와 LeakReLU에 의해 모델의 수렴속도가 향상되었다.
# 배치 정규화와 드롭 아웃던분에 모델의 성능이 빠르게 향상되었음을 관찰 할 수 있다.

# 인공 신경망은 nn.Linear를 통해서 전결합층을 구성하고, 
# nn.ReLU와 nn.LeakReRU로 활성화 함수를 구현했다. 
# 그리고 modle(input)과 loss.backward()로 순전파와 역전파를 구현했다. 
# optimizer.step을 통해서 가중치를 업데이트한다.

# 합성곱 신경망의 합성곱 계층은 nn.Conv2d를 사용해서 만들었고, 
# 패딩과 스트라이드를 명시적으로 설정하여 특정 맵의 크기를 조절하였다. 
# nn.MaxPool2d를 사용해서 중요 특징을 유지하면서 차원을 축소하였다. 
# nn.Flatten이후에 nn.Linear로 이어지게 구조를 만들었다.

# 드롭 아웃은 nn.Dropout(0.5)를 써서 과적합을 방지했다.
# 배치 정규화는 nn.BatchNorm2d를 합성곱 계층 뒤에 배치해서 학습 안정성을 높였다.
# 옵티마이져는 adam을 사용했다.
