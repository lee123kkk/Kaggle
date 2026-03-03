import os
import random
import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------
# 1. 시드값 고정 (재현성 확보)
# ---------------------------------------------------------
def seed_everything(seed=50):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

# ---------------------------------------------------------
# 2. 커스텀 데이터셋 클래스 정의
# ---------------------------------------------------------
class ImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        super().__init__()
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]    # 이미지 ID
        img_path = os.path.join(self.img_dir, img_id) # 이미지 파일 절대 경로
        
        image = cv2.imread(img_path)     # 이미지 파일 읽기 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정
        
        # 테스트 데이터셋은 타깃값이 없으므로 예외 처리
        if len(self.df.columns) > 1:
            label = self.df.iloc[idx, 1]
        else:
            label = 0

        if self.transform is not None:
            image = self.transform(image) # 변환기가 있다면 이미지 변환
            
        return image, label

# ---------------------------------------------------------
# 3. 개선된 모델 아키텍처 정의 (더 깊고 안정적임)
# ---------------------------------------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__() 
        # 1 ~ 5번째 {합성곱, 배치 정규화, 활성화 함수, 최대 풀링} 계층 
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=2),
                                    nn.BatchNorm2d(32),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=2),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=2),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        
        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=2),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        
        self.layer5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=2),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(),
                                    nn.MaxPool2d(kernel_size=2))
        
        # 평균 풀링 계층 
        self.avg_pool = nn.AvgPool2d(kernel_size=4) 
        # 전결합 계층
        self.fc1 = nn.Linear(in_features=512 * 1 * 1, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg_pool(x)
        x = x.view(-1, 512 * 1 * 1) # 평탄화
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# ---------------------------------------------------------
# 4. 메인 실행 함수
# ---------------------------------------------------------
def main():
    print("="*50)
    print("개선된 선인장 식별 모델 학습을 시작합니다. (Epoch 70)")
    
    seed_everything(50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 사용 중인 장비: {device}")
    print("="*50)

    # 데이터 경로 설정 (WSL 절대 경로)
    base_path = '/home/lee123kkk/kaggle_workspace/Competition/Aerial_Cactus_Identification/'
    train_dir = os.path.join(base_path, 'train/train/')
    test_dir = os.path.join(base_path, 'test/test/')
    
    labels = pd.read_csv(os.path.join(base_path, 'train.csv'))
    submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

    # 훈련/검증 분리
    train, valid = train_test_split(labels, 
                                    test_size=0.1,
                                    stratify=labels['has_cactus'],
                                    random_state=50)

    # ★ 추가된 부분: 데이터 증강(Augmentation) 및 정규화(Normalization)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(32, padding_mode='symmetric'),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(32, padding_mode='symmetric'),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    dataset_train = ImageDataset(df=train, img_dir=train_dir, transform=transform_train)
    dataset_valid = ImageDataset(df=valid, img_dir=train_dir, transform=transform_test)

    loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 초기화
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.00006)

    # ---------------------------------------------------------
    # 모델 학습
    # ---------------------------------------------------------
    epochs = 70
    print("\n[훈련 시작]")
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0 
        
        for images, targets in loader_train:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            epoch_loss += loss.item() 
            loss.backward()
            optimizer.step()
            
        print(f'에폭 [{epoch+1}/{epochs}] - 평균 손실값: {epoch_loss/len(loader_train):.4f}')

    # ---------------------------------------------------------
    # 검증 데이터 평가
    # ---------------------------------------------------------
    print("\n[검증 평가 시작]")
    model.eval() 
    
    true_list = []
    preds_list = []

    with torch.no_grad(): 
        for images, targets in loader_valid:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            preds = torch.softmax(outputs.cpu(), dim=1)[:, 1] 
            true = targets.cpu() 
            
            preds_list.extend(preds.numpy())
            true_list.extend(true.numpy())
            
    roc_auc = roc_auc_score(true_list, preds_list)
    print(f'검증 데이터 ROC AUC 점수: {roc_auc:.6f}')

    # ---------------------------------------------------------
    # 테스트 예측 및 제출 파일 저장
    # ---------------------------------------------------------
    print("\n[테스트 데이터 예측 시작]")
    dataset_test = ImageDataset(df=submission, img_dir=test_dir, transform=transform_test)
    loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)

    model.eval() 
    preds = [] 

    with torch.no_grad(): 
        for images, _ in loader_test:
            images = images.to(device)
            
            outputs = model(images)
            preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
            preds.extend(preds_part)

    submission['has_cactus'] = preds
    output_csv_path = os.path.join(base_path, 'submission_advanced.csv')
    submission.to_csv(output_csv_path, index=False)
    print(f"\n최종 제출 파일이 생성되었습니다: {output_csv_path}")

if __name__ == "__main__":
    main()
#==========================================================================
# 선인장 탐지 개선1
 # 베이스라인 당시 0.9902%였던 ROC_ACU점수가 0.999827까지 올라갔다.

 # 핵심 개선 사항
     # 이미지를 무작위로 뒤집고, 최대 10도씩 비틀어서 이미지의 크기를 수십 배로 뻥튀기했다.
     # 픽셀값을 정규화하여 신경망이 안정적으로 가중치를 찾을 수 있도록 헀다.
     # 합성곱 층이 2개에서 5개로 깊어졌다.
     # ReLU대신 LeakyReLU를 사용해서 Dying ReLU현상을 방지했다.
     # SGD 대신에 Adamax를 사용했다.
     # 학습률을 0.01에서 0.00006으로 극단적으로 낮추고 에폭을 70으로 크게 늘렸다.
