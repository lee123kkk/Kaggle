import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2
import torch # GPU 확인을 위해 PyTorch 임포트

def main():
    # 1. GPU 사용 가능 여부 확인
    print("="*40)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"GPU가 정상적으로 인식되었습니다. 사용 디바이스: {device}")
        print(f"GPU 모델명: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"GPU를 찾을 수 없습니다. 사용 디바이스: {device}")
    print("="*40)

    # 2. 데이터 경로 설정 (WSL 내부 절대 경로 사용)
    base_path = '/home/lee123kkk/kaggle_workspace/Competition/Aerial_Cactus_Identification/'
    train_dir = os.path.join(base_path, 'train/train/')
    test_dir = os.path.join(base_path, 'test/test/')
    
    # CSV 파일 경로 (train 폴더 바깥에 있다고 가정)
    train_csv_path = os.path.join(base_path, 'train.csv')
    submission_csv_path = os.path.join(base_path, 'sample_submission.csv')

    # 데이터 로드
    labels = pd.read_csv(train_csv_path)
    submission = pd.read_csv(submission_csv_path)

    print("\n[훈련 데이터 CSV 상위 5개]")
    print(labels.head())

    # 3. 타깃값 분포 파이 그래프 그리기
    mpl.rc('font', size=15)
    plt.figure(figsize=(7, 7))

    label = ['Has cactus', "Hasn't cactus"] # 타깃값 레이블
    # 타깃값 분포 파이 그래프
    plt.pie(labels['has_cactus'].value_counts(), labels=label, autopct='%.1f%%')
    plt.title("Cactus Distribution")
    plt.show() # 파이썬 스크립트에서는 plt.show()를 호출해야 창이 뜹니다.

    # 4. 이미지 데이터 개수 확인
    num_train = len(os.listdir(train_dir))
    num_test = len(os.listdir(test_dir))

    print(f'\n훈련 데이터 개수: {num_train}')
    print(f'테스트 데이터 개수: {num_test}')

    # 5. 선인장을 포함하는/포함하지 않는 이미지 출력
    mpl.rc('font', size=7)
    
    # --- 선인장을 포함하는 이미지 출력 ---
    plt.figure(figsize=(15, 6))    # 전체 Figure 크기 설정
    grid = gridspec.GridSpec(2, 6) # 서브플롯 배치(2행 6열로 출력)
        
    # 선인장을 포함하는 이미지 파일명(마지막 12개) 
    last_has_cactus_img_name = labels[labels['has_cactus']==1]['id'][-12:]

    # 이미지 출력 
    print("\n선인장이 포함된 이미지 로딩 중...")
    for idx, img_name in enumerate(last_has_cactus_img_name):
        img_path = os.path.join(train_dir, img_name)   # 이미지 파일 경로 
        image = cv2.imread(img_path)                   # 이미지 파일 읽기 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정 
        ax = plt.subplot(grid[idx])
        ax.imshow(image)                               # 이미지 출력 
        ax.axis('off')                                 # 축 제거 (깔끔하게 보기 위함)
    
    plt.suptitle("Images WITH Cactus")
    plt.show()

    # --- 선인장을 포함하지 않는 이미지 출력 ---
    plt.figure(figsize=(15, 6))    # 전체 Figure 크기 설정
    grid = gridspec.GridSpec(2, 6) # 서브플롯 배치
        
    # 선인장을 포함하지 않는 이미지 파일명(마지막 12개) 
    last_hasnt_cactus_img_name = labels[labels['has_cactus']==0]['id'][-12:]

    # 이미지 출력 
    print("선인장이 포함되지 않은 이미지 로딩 중...")
    for idx, img_name in enumerate(last_hasnt_cactus_img_name):
        img_path = os.path.join(train_dir, img_name)   # 이미지 파일 경로
        image = cv2.imread(img_path)                   # 이미지 파일 읽기
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # 이미지 색상 보정
        ax = plt.subplot(grid[idx])
        ax.imshow(image)                               # 이미지 출력 
        ax.axis('off')                                 # 축 제거
    
    plt.suptitle("Images WITHOUT Cactus")
    plt.show()

    print(f"\n마지막으로 읽은 이미지의 형태(Shape): {image.shape}")

if __name__ == "__main__":
    main()
#==========================================================================
# 선인장 파일 전처리 및 데이터 증강

# 데이터 형태 및 크기 분석
    # 이미지의 형태가 (32, 32, 3)로 작은 컬러 이미지인것을 알 수 있다.
    # 이미지가 매우 작기 때문에 더 줄이면 선인장의 특징이 사라질 수 있다. 
    # 그러므로 기본적으로는 이 크기를 유지하고, ResNet등 무거운 모델을 전이 학습으로 사용할 떄는 모델 입력 크기에 맞춰 강제로 늘려야 할 수 도 있다.
    # 이미지를 0에서 1사리로 스케일링하고 평균을 0, 표준편차를 1로 맞추는 정규화 작업이 필수적이다.

# 클래스 분포 분석
    # 선인장이 있는 이미지가 75.1%이고 없는 이미지가 24.9%이다. 클래스 불균형이 존재하는 것을 학인할 수 있다. 
    # 클래스 불균형이 존재하는 것을 학인할 수 있다. 
    # 단순히 정확도만 보면 무조건 선인장이 있다고 해도 75점 이상 나오므로 AUC-ROC 커브나 F1_score같은 지표를 함께 봐야 한다. 
    # 선인장이 없는 데이터를 증강 기법을 통해서 인위적으로 늘려서 비율을 맞출 수 있다.

# 시각적 특징 분석
    # 항공 사진이므로 위에서 내려다 본 시점이고, 선인장은 놓여 있는 방향이가 각도가 제각각이다. 이미지 내에서 선인장이 중앙에 있기도 하고 구석에 쏠려 있기도 한다. 
    # 항공 사진은 위아래가 없기 때문에 이미지를 마음대로 돌려도 자연스러운 지형이 된다. 
    # 따라서 좌우/상하 반전을 시킨다거나 무작위로 회전시키는 등의 데이터 증강이 가능하다.

# 픽셀 값을 정규화하고, 훈련 데이터에 상하/좌우  반전 및 회전 증강 기법을 적용시킨다. 

