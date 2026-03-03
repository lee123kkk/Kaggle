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
    random.seed(seed)                # 파이썬 난수 생성기 시드 고정
    np.random.seed(seed)             # 넘파이 난수 생성기 시드 고정
    torch.manual_seed(seed)          # 파이토치 난수 생성기 시드 고정 (CPU 사용 시)
    torch.cuda.manual_seed(seed)     # 파이토치 난수 생성기 시드 고정 (GPU 사용 시)
    torch.cuda.manual_seed_all(seed) # 파이토치 난수 생성기 시드 고정 (멀티GPU 사용 시)
    torch.backends.cudnn.deterministic = True # 확정적 연산 사용
    torch.backends.cudnn.benchmark = False    # 벤치마크 기능 해제

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
        # img_dir과 img_id를 결합하여 절대 경로 생성
        img_path = os.path.join(self.img_dir, img_id)
        
        image = cv2.imread(img_path)     # 이미지 파일 읽기 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정
        
        # 테스트 데이터셋(submission)은 타깃값이 없으므로 예외 처리
        if len(self.df.columns) > 1:
            label = self.df.iloc[idx, 1] # 훈련 데이터: 레이블 있음
        else:
            label = 0                    # 테스트 데이터: 임의의 0 반환

        if self.transform is not None:
            image = self.transform(image) # 변환기가 있다면 이미지 변환
            
        return image, label

# ---------------------------------------------------------
# 3. 모델 아키텍처 정의
# ---------------------------------------------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__() 
        # 첫 번째 합성곱 계층 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2) 
        # 두 번째 합성곱 계층 
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2) 
        # 최대 풀링 계층 
        self.max_pool = nn.MaxPool2d(kernel_size=2) 
        # 평균 풀링 계층 
        self.avg_pool = nn.AvgPool2d(kernel_size=2) 
        # 전결합 계층 
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)
        
    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4) # 평탄화
        x = self.fc(x)
        return x

# ---------------------------------------------------------
# 4. 메인 실행 함수
# ---------------------------------------------------------
def main():
    print("="*50)
    print("선인장 식별 모델 학습을 시작합니다.")
    
    seed_everything(50)

    # 장비 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"현재 사용 중인 장비: {device}")
    print("="*50)

    # 데이터 경로 설정 (WSL 절대 경로)
    base_path = '/home/lee123kkk/kaggle_workspace/Competition/Aerial_Cactus_Identification/'
    train_dir = os.path.join(base_path, 'train/train/')
    test_dir = os.path.join(base_path, 'test/test/')
    
    labels_path = os.path.join(base_path, 'train.csv')
    submission_path = os.path.join(base_path, 'sample_submission.csv')

    # CSV 데이터 로드
    labels = pd.read_csv(labels_path)
    submission = pd.read_csv(submission_path)

    # 훈련 데이터 / 검증 데이터 분리 (9:1)
    train, valid = train_test_split(labels, 
                                    test_size=0.1,
                                    stratify=labels['has_cactus'],
                                    random_state=50)

    print(f'훈련 데이터 개수: {len(train)}')
    print(f'검증 데이터 개수: {len(valid)}')

    # 데이터셋 및 데이터로더 생성
    transform = transforms.ToTensor()

    dataset_train = ImageDataset(df=train, img_dir=train_dir, transform=transform)
    dataset_valid = ImageDataset(df=valid, img_dir=train_dir, transform=transform)

    loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 초기화
    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # ---------------------------------------------------------
    # 모델 학습
    # ---------------------------------------------------------
    epochs = 10 # 총 에폭
    print("\n[훈련 시작]")
    for epoch in range(epochs):
        model.train()  # 모델을 훈련 상태로 설정
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
            
        print(f'에폭 [{epoch+1}/{epochs}] - 손실값(Loss): {epoch_loss/len(loader_train):.4f}')

    # ---------------------------------------------------------
    # 검증 데이터 평가
    # ---------------------------------------------------------
    print("\n[검증 평가 시작]")
    model.eval() 
    
    preds_list = [] # ★ 누락되었던 리스트 초기화 추가
    true_list = []  # ★ 누락되었던 리스트 초기화 추가

    with torch.no_grad(): 
        for images, targets in loader_valid:
            images = images.to(device)
            targets = targets.to(device) 
            
            outputs = model(images)
            preds = torch.softmax(outputs.cpu(), dim=1)[:, 1] # 양성(선인장) 클래스 예측 확률  
            true = targets.cpu() 
            
            preds_list.extend(preds.numpy()) # 리스트에 결과 추가
            true_list.extend(true.numpy())
            
    # 검증 데이터 ROC AUC 점수 계산
    roc_auc = roc_auc_score(true_list, preds_list)
    print(f'검증 데이터 ROC AUC 점수: {roc_auc:.4f}')

    # ---------------------------------------------------------
    # 테스트 데이터 예측 및 제출 파일 생성
    # ---------------------------------------------------------
    print("\n[테스트 데이터 예측 시작]")
    dataset_test = ImageDataset(df=submission, img_dir=test_dir, transform=transform)
    loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)

    model.eval() 
    preds = [] 

    with torch.no_grad(): 
        for images, _ in loader_test:
            images = images.to(device)
            
            outputs = model(images)
            # 타깃값이 1일 확률(예측값) 추출
            preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
            preds.extend(preds_part)

    # 제출용 파일 생성
    submission['has_cactus'] = preds
    output_csv_path = os.path.join(base_path, 'submission.csv')
    submission.to_csv(output_csv_path, index=False)
    print(f"\n최종 제출 파일이 생성되었습니다: {output_csv_path}")

if __name__ == "__main__":
    main()

#============================================================================
# 선인장 인식 baseline

 # 베이스라인 모델에서 수행된 작업
    # 데이터 분할을 통해서 EDA에서 확인했던 3:1비율을 훈련용 데이터와 검증용 데이터에서로 똑같이 유지되도록 분할했다.
    # 이미지를 텐서로 바꾸고 픽셀값을 0과 1사이로 바꾸는 기초적인 데이터 전처리만 수행했다.
    # 모델 구조는 합성곱2개, 풀링층3개, 전결합층 1개로 구성된 단순한 신경망을 사용했다.
    # 최적화에는 SGD를 사용헀고, 학습률은 0.01로 고정하였다.

 # 성능 개선 방안   
    # 무작위 상하/좌우 반전, 무작위 회전등을 통해서 과적합을 방지한다.
    # 이미지 정규화를 추가한다.
    # 전이 학습을 도입한다.

# 베이스라인 코드를 확인해보고 더 발전시키기 위해서 필요한 것들을 분석할 수 있다.