#9_Transfer_Learning

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights # 최신 버전 가중치 로드 방식
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# 1. 설정 및 데이터 전처리
# ==========================================
# [설정] GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# [텐서보드] 로그 저장 경로 설정
writer = SummaryWriter('runs/cifar10_transfer_learning_resnet18')

# [전처리] 전이 학습을 위한 핵심 단계
# 사전 학습된 모델(ResNet)은 ImageNet 데이터의 평균과 표준편차로 학습되었으므로,
# 입력 데이터도 이와 동일하게 정규화해주는 것이 성능에 매우 중요합니다.
stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # ImageNet의 Mean, Std

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),      # ResNet의 기본 입력 크기로 조정 (선택사항이나 성능 향상 도움)
    transforms.RandomHorizontalFlip(),  # [성능] 데이터 증강 (Augmentation)
    transforms.ToTensor(),
    transforms.Normalize(*stats)        # [성능] 배치 정규화 입력 전 스케일 조정
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(*stats)
])

# 데이터셋 로드
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==========================================
# 2. [전이 학습] 모델 로드 및 수정 (핵심 파트)
# ==========================================
print("Loading Pre-trained ResNet18...")

# [개념: 전이 학습] 사전 학습된 가중치(Weights)를 가져옵니다.
# 'DEFAULT'는 가장 성능이 좋은 가중치를 의미합니다.
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# [개념: 특징 추출기 고정 (Feature Extractor Freezing)]
# 이미 학습된 CNN 부분(합성곱, 풀링, 배치정규화 등)의 가중치는 변경되지 않도록 고정합니다.
# 이렇게 하면 학습 속도가 빨라지고, 적은 데이터로도 좋은 성능을 냅니다.
for param in model.parameters():
    param.requires_grad = False

# [개념: 모델 구조 수정]
# ResNet18의 마지막 계층(fc)은 원래 1000개의 클래스(ImageNet)를 분류합니다.
# 이를 CIFAR-10의 10개 클래스에 맞게 교체합니다.
# 새로 생성된 레이어는 기본적으로 requires_grad=True이므로 이 부분만 학습됩니다.
num_ftrs = model.fc.in_features # 512개
model.fc = nn.Linear(num_ftrs, 10) # [ANN] 최종 분류기 (퍼셉트론)

model = model.to(device)

# [텐서보드] 모델 그래프 기록 (임의의 입력 데이터 주입)
sample_images, _ = next(iter(train_loader))
writer.add_graph(model, sample_images.to(device))

# ==========================================
# 3. 손실 함수 및 옵티마이저
# ==========================================
criterion = nn.CrossEntropyLoss()

# [개념: 옵티마이저]
# 전체 모델 파라미터가 아니라, 우리가 교체한 'model.fc'의 파라미터만 최적화합니다.
# 이미지 속 개념: 모멘텀(Momentum)이 포함된 SGD 사용 (혹은 Adam 사용 가능)
optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

# ==========================================
# 4. 학습 루프 (Training Loop)
# ==========================================
print("Starting Transfer Learning...")

EPOCHS = 3 # 전이 학습은 수렴이 매우 빠르므로 적은 에포크로도 충분합니다.

for epoch in range(EPOCHS):
    running_loss = 0.0
    correct = 0
    total = 0
    
    # [주의] 전이 학습 시, model.train()을 호출하더라도 
    # 고정된(Frozen) 레이어의 배치 정규화(Batch Norm) 통계치는 
    # 사전 학습된 상태를 유지하는 것이 일반적입니다 (eval 모드 성격 유지).
    # 하지만 PyTorch의 기본 동작을 위해 여기서는 train()을 선언하되, 
    # 앞서 requires_grad=False로 가중치 업데이트만 막았습니다.
    model.train() 

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # [ANN] 경사 하강법 단계
        optimizer.zero_grad()
        
        # [ANN] 순전파 (Forward)
        # ResNet 내부: Conv -> BN -> ReLU -> Pooling -> ... -> FC 순으로 진행
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # [ANN] 역전파 (Backward) 및 가중치 갱신
        loss.backward()
        optimizer.step()

        # 통계 계산
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 100 배치마다 로그 출력
        if i % 100 == 99:
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

    # [텐서보드] 에포크별 성능 기록
    epoch_acc = 100 * correct / total
    writer.add_scalar('Loss/train', running_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    print(f"Epoch {epoch+1} Accuracy: {epoch_acc:.2f}%")

print("Transfer Learning Finished.")


# ==========================================
# 5. 최종 성능 평가 및 텐서보드 종료
# ==========================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

final_acc = 100 * correct / total
print(f'Final Accuracy on CIFAR-10 test set: {final_acc:.2f}%')

# [텐서보드] 하이퍼파라미터와 최종 성능 기록
writer.add_hparams(
    {'lr': 0.001, 'batch_size': 32, 'model': 'ResNet18-Transfer'},
    {'hparam/accuracy': final_acc}
)

writer.close()
#====================================================
# 첫번째 에포크의 시작부분에서는 손실이 높았지만, 배치가 500을 넘어가면서 0.866으로 떨어졌다.
# 단 3번의 에포크 만에 최종 정확도가 80.2%에 도달했다
# accuracy그래프가 처음부터 70%이상에서 출발하여 안정적으로 평행을 유지한다.
# 이를 통해서 안정적으로 학습을 하고 있음을 알 수 있다.

# 전이 학습을 통해서 빠르고 안정적이게 성능을 얻을 수 있다.
