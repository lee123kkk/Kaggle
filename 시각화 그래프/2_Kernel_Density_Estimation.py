# 2_Kernel_Density_Estimation
# 이미지가 너무 어둡거나 밝지는 않은지 RGB 채널별로 분포가 치우쳐져 있지 않은지 확인한다.
# normalization의 핵심 근거가 된다.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision

# [수정된 부분] 0. 라이브러리 및 데이터셋 다시 로드
# 이전 파일과 별개의 스크립트이므로 train_ds를 다시 정의해야 합니다.
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 1. 전체 데이터를 하나의 거대한 배열로 변환 (분석용)
# PyTorch 텐서로 바꾸기 전에 Numpy 배열 상태에서 픽셀값을 0~1 사이로 스케일링합니다.
data = train_ds.data # (50000, 32, 32, 3) 형태의 numpy array
data = data / 255.0  

# 2. 채널별(R, G, B)로 데이터 분리 및 1차원으로 펼치기
# [:, :, :, 0]은 모든 이미지, 모든 가로, 모든 세로의 첫 번째 채널(Red)을 의미합니다.
# flatten()을 통해 그래프를 그리기 쉽도록 1차원 배열로 평탄화합니다.
r_vals = data[:, :, :, 0].flatten()
g_vals = data[:, :, :, 1].flatten()
b_vals = data[:, :, :, 2].flatten()

# 3. 커널 밀도 추정(KDE) 시각화
# KDE(Kernel Density Estimation)는 히스토그램을 부드러운 곡선으로 만들어 데이터의 분포를 보여줍니다.
plt.figure(figsize=(12, 5))
sns.kdeplot(r_vals, color='red', label='Red Channel', fill=True, alpha=0.1)
sns.kdeplot(g_vals, color='green', label='Green Channel', fill=True, alpha=0.1)
sns.kdeplot(b_vals, color='blue', label='Blue Channel', fill=True, alpha=0.1)

# 그래프 꾸미기
plt.title("Pixel Intensity Distribution (R, G, B)")
plt.xlabel("Pixel Value (Normalized 0~1)")
plt.ylabel("Density")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 4. 평균과 표준편차 계산 (전처리 수치 확보)
# 이 수치들이 바로 PyTorch 모델에 이미지를 넣기 전 정규화(Normalization)에 사용될 핵심 값입니다.
print(f"Mean - R: {r_vals.mean():.4f}, G: {g_vals.mean():.4f}, B: {b_vals.mean():.4f}")
print(f"Std  - R: {r_vals.std():.4f},  G: {g_vals.std():.4f},  B: {b_vals.std():.4f}")
#=====================================================================
# 

# 이미지가 너무 어둡거나 밝지는 않은지 RGB 채널별로 분포가 치우쳐져 있지 않은지 확인한다.
# normalization의 핵심 근거가 된다.

# 평균값을 보면 red와 green의 픽셀 강도가 blue보다 높다. 붉은색, 초록색 톤이 더 많이 포함되어 있다.
# 분포의 형태를 보면 정규화 전 이미지는 0~255사이로 들쭉날쭉하다.
# 채널별로 평균과 분산이 다르기 때문에 평균을 0, 표중편차를 1로 마추는 표준화 작업이 필요하다.
