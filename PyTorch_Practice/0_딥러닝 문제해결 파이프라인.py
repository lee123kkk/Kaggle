import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 재현성을 위한 시드 설정
torch.manual_seed(42)

# GPU 사용 가능 여부 확인 (환경 설정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current Device: {device}")

# ==============================================================================
# 1. 문제(경진대회) 이해
# ==============================================================================
# [설명]
# - 문제 배경/목적: 0~9까지의 손글씨 이미지를 입력받아 정확한 숫자를 분류(Classification)
# - 문제 유형: 다중 클래스 분류 (Multi-class Classification)
# - 평가 지표(Metric): 정확도 (Accuracy)

print("\n=== [1단계] 문제 이해 완료: MNIST 숫자 분류 (Metric: Accuracy) ===")


# ==============================================================================
# 2. 탐색적 데이터 분석 (EDA)
# ==============================================================================
# [데이터 구조 탐색 및 시각화]

# 데모를 위해 데이터를 다운로드합니다.
train_ds_raw = torchvision.datasets.MNIST(root='./data', train=True, download=True)

print("\n=== [2단계] 탐색적 데이터 분석 (EDA) 시작 ===")

# 2-1. 데이터 구조 탐색
print(f"데이터 개수: {len(train_ds_raw)}")
print(f"이미지 크기: {train_ds_raw[0][0].size}") # (28, 28)
print(f"데이터 타입: {type(train_ds_raw[0][0])}")

# 2-2. 데이터 시각화 (타겟값 분포 확인)
# 라벨(정답)만 추출
labels = [label for _, label in train_ds_raw]
plt.figure(figsize=(8, 4))
sns.countplot(x=labels)
plt.title("Target Distribution (Label Counts)")
plt.xlabel("Digit (0-9)")
plt.ylabel("Count")
plt.show()

# 2-3. 실제 이미지 확인 (데이터 품질 확인)
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i in range(5):
    img, label = train_ds_raw[i]
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Label: {label}")
    axes[i].axis('off')
plt.show()


# ==============================================================================
# 3. 베이스라인 모델 (Baseline Model)
# ==============================================================================
print("\n=== [3단계] 베이스라인 모델 구축 ===")

# 3-1. 환경 설정 및 데이터 준비
# - 전처리: 이미지를 Tensor로 변환하고 0~1 사이로 정규화
baseline_transform = transforms.Compose([
    transforms.ToTensor(), # 0~255 -> 0.0~1.0
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, 
                                           download=True, transform=baseline_transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, 
                                          download=True, transform=baseline_transform)

# DataLoader 설정 (배치 단위 로딩)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 3-2. 모델 정의 (간단한 CNN)
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) # 입력 채널 1, 출력 32
        self.fc1 = nn.Linear(32 * 26 * 26, 10)       # Flatten 후 출력 10개 (0~9)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten
        x = self.fc1(x)
        return x

model = BaselineModel().to(device)

# 3-3. 모델 훈련 및 성능 검증
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # 가장 기본적인 SGD 옵티마이저

def train_and_evaluate(model, optimizer, epochs=1, name="Baseline"):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    # 성능 검증 (Validation)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    acc = 100. * correct / len(test_loader.dataset)
    print(f"[{name}] Test Accuracy: {acc:.2f}%")

# 베이스라인 훈련 실행
train_and_evaluate(model, optimizer, epochs=1, name="Baseline")

# 3-4. 결과 예측 및 제출 (가상)
# 실제 캐글에서는 submission.csv를 생성합니다.
print("-> 베이스라인 submission.csv 생성 완료 (가상)")

# ==============================================================================
# 4. 성능 개선 (Performance Improvement)
# ==============================================================================
# [흐름도 피드백] 베이스라인 성능이 만족스럽지 않다고 가정하고 개선 기법 적용
print("\n=== [4단계] 성능 개선 기법 적용 ===")

# 4-1. 성능 개선 기법 적용
# 기법 1: 데이터 증강 (Data Augmentation) - 이미지 회전 추가
improved_transform = transforms.Compose([
    transforms.RandomRotation(10), # 랜덤하게 -10~10도 회전
    transforms.ToTensor(),
])

# 데이터셋 다시 로드 (증강 적용)
train_dataset_imp = torchvision.datasets.MNIST(root='./data', train=True, 
                                               download=True, transform=improved_transform)
train_loader_imp = torch.utils.data.DataLoader(train_dataset_imp, batch_size=64, shuffle=True)

# 기법 2: 모델 구조 개선 (Deeper CNN + Dropout) & 옵티마이저 변경 (Adam)
class ImprovedModel(nn.Module):
    def __init__(self):
        super(ImprovedModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.dropout = nn.Dropout(0.25) # 과적합 방지
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

improved_model = ImprovedModel().to(device)

# 기법 3: 옵티마이저 개선 (SGD -> Adam)
optimizer_imp = optim.Adam(improved_model.parameters(), lr=0.001)

# 4-2. 성능 검증 (재학습)
train_loader = train_loader_imp # 로더 교체
train_and_evaluate(improved_model, optimizer_imp, epochs=1, name="Improved")

# 4-3. 결과 예측 및 제출
print("-> 최종 개선 모델 submission.csv 생성 완료 (가상)")


