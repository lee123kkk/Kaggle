# Cactus_CNN

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==========================================
# 1. 원본 2D 이미지 시뮬레이션 (선인장 줄기와 가지)
# ==========================================
# 10x10 픽셀의 단순화된 십자가 형태(선인장) 생성
img = np.zeros((10, 10))
img[2:8, 4:6] = 1.0  # 수직 줄기 부분
img[4:6, 2:8] = 1.0  # 수평 가지 부분

# ==========================================
# 2. 평탄화(Flatten) 시뮬레이션
# ==========================================
# 일반 전결합층(Fully Connected)에 넣기 위해 1D로 펴버림 (1x100 배열)
img_flatten = img.flatten().reshape(1, -1)

# ==========================================
# 3. 합성곱(Convolution) 시뮬레이션
# ==========================================
# 수직선(가시나 줄기의 좌우 경계선)을 찾아내는 3x3 특징 추출 커널
filter_kernel = np.array([[-1, 0, 1],
                          [-1, 0, 1],
                          [-1, 0, 1]])
# 이미지 위를 슬라이딩하며 연산 (특징 맵 생성)
img_conv = convolve2d(img, filter_kernel, mode='same')

# ==========================================
# 4. 활성화 함수 (ReLU vs Leaky ReLU) 시뮬레이션
# ==========================================
x = np.linspace(-5, 5, 100)
relu = np.maximum(0, x)
leaky_relu = np.where(x > 0, x, 0.1 * x)  # 음수 구간에 0.1의 기울기 부여

# ==========================================
# 5. 시각화 (Explore & Discover)
# ==========================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# [Plot 1] 원본 2D 이미지 보존
axes[0, 0].imshow(img, cmap='Greens', interpolation='nearest')
axes[0, 0].set_title("1. Original 2D Image (Cactus Shape)\n[Spatial Information Intact]", fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(range(10))
axes[0, 0].set_yticks(range(10))
axes[0, 0].grid(color='white', linestyle='-', linewidth=1)

# [Plot 2] 형태 파괴 (Flatten)
axes[0, 1].imshow(img_flatten, cmap='Greens', aspect=5, interpolation='nearest')
axes[0, 1].set_title("2. Flattened 1D Array\n[Spatial Information Destroyed]", fontsize=14, fontweight='bold', color='red')
axes[0, 1].set_yticks([])
axes[0, 1].set_xlabel("Pixel Index (0 ~ 99)")

# [Plot 3] 형태 보존 및 특징 추출 (Convolution)
axes[1, 0].imshow(img_conv, cmap='RdBu', interpolation='nearest')
axes[1, 0].set_title("3. Output after CNN Filter\n[Extracts Vertical Edges / Thorns]", fontsize=14, fontweight='bold', color='blue')
axes[1, 0].set_xticks(range(10))
axes[1, 0].set_yticks(range(10))
axes[1, 0].grid(color='white', linestyle='-', linewidth=1)

# [Plot 4] 신경망의 생명줄 (Leaky ReLU)
axes[1, 1].plot(x, relu, label='ReLU', lw=4, color='dodgerblue')
axes[1, 1].plot(x, leaky_relu, label='Leaky ReLU', lw=4, linestyle='--', color='darkorange')
axes[1, 1].axhline(0, color='gray', lw=1.5, linestyle=':')
axes[1, 1].axvline(0, color='gray', lw=1.5, linestyle=':')
axes[1, 1].set_title("4. Activation Function\n[Dying ReLU vs Leaky ReLU]", fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==========================================
# 6. 터미널 출력 (강의 진행 시나리오)
# ==========================================
print("\n" + "="*70)
print("🧠 [탐구 미션] 선인장의 가시를 알아보는 신경망 구조 설계")
print("="*70)
print("Q1. [Plot 1]의 십자가 모양 선인장 이미지를 일반 인공신경망에 넣기 위해")
print("    [Plot 2]처럼 1열로 쫙 펴버리면(Flatten) 무슨 일이 발생할까요?")
print("    -> (관찰) 위아래 픽셀 간의 연결성(모양, 형태)이 완전히 파괴되어 버립니다!")
print("-" * 70)
print("Q2. 그렇다면 2D 공간 정보를 유지하면서 특징을 뽑아내려면 어떻게 해야 할까요?")
print("    -> (해결) [Plot 3]처럼 돋보기(필터)를 대고 훑는 합성곱(Convolution)을 사용합니다.")
print("    -> (관찰) 붉은색과 푸른색으로 특정 패턴(수직선 경계)만 선명하게 추출되었습니다!")
print("-" * 70)
print("Q3. 망을 깊게 쌓다가 특정 층에서 계산 값이 음수(-3)로 떨어졌습니다.")
print("    [Plot 4]에서 일반 'ReLU(파란선)'를 통과시키면 어떻게 될까요?")
print("    -> (위기) 무조건 0이 되어버립니다. 뉴런이 죽어버려서(Dying ReLU) 더 이상 학습을 못 합니다.")
print("    -> (해결) 이때 'Leaky ReLU(주황 점선)'를 쓰면 0이 아니라 -0.3이라는 미세한 생명줄을 남겨")
print("       깊은 망에서도 기울기 소실(Gradient Vanishing) 없이 무사히 학습을 이어갈 수 있습니다!")
print("="*70 + "\n")
