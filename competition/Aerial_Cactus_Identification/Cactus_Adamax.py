# Cactus_Adamax

import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 훈련 과정 시뮬레이션 (Learning Rate에 따른 Loss 곡선)
# ==========================================
epochs = np.arange(1, 51)

# [상황 A] LR이 너무 작을 때 (기어가듯 느림)
loss_crawling = 2.0 * np.exp(-0.02 * epochs)
# [상황 B] LR이 적절할 때 (안정적 수렴)
loss_optimal = 2.0 * np.exp(-0.15 * epochs)
# [상황 C] LR이 너무 클 때 (발산/진동) - 노이즈 추가하여 시뮬레이션
np.random.seed(42)
loss_exploding = 1.0 + np.exp(0.04 * epochs) + np.random.normal(0, 0.4, 50)

# ==========================================
# 2. 옵티마이저 등반 경로 시뮬레이션 (SGD vs Adamax)
# 손실 함수 지형(Loss Landscape) 정의: 계곡 형태 (y축 경사가 훨씬 가파름)
# ==========================================
def loss_function(x, y):
    return x**2 + 5 * y**2

def gradient(x, y):
    return np.array([2*x, 10*y])

# 시작점 세팅 (계곡의 가장자리 높은 곳)
start_pt = np.array([-4.0, 3.0])
steps = 20

# 2-1. 순수 SGD (확률적 경사 하강법) 경로
path_sgd = [start_pt]
pt = start_pt.copy()
lr_sgd = 0.17 # 약간 큰 보폭 설정 (요동치게 만듦)
for _ in range(steps):
    grad = gradient(pt[0], pt[1])
    pt = pt - lr_sgd * grad
    path_sgd.append(pt)
path_sgd = np.array(path_sgd)

# 2-2. Adamax (적응형 학습률 + 관성) 경로 시뮬레이션
# (데모를 위한 간략화된 알고리즘 구현)
path_adamax = [start_pt]
pt = start_pt.copy()
lr_adamax = 0.7 # Adam 계열은 스텝 사이즈를 알아서 조절하므로 초기값이 큼
m = np.zeros(2)
u = np.zeros(2)
beta1, beta2 = 0.9, 0.999
for t in range(1, steps + 1):
    grad = gradient(pt[0], pt[1])
    # 관성(Momentum) 계산
    m = beta1 * m + (1 - beta1) * grad
    # 최대 그레디언트(Infinity norm) 업데이트
    u = np.maximum(beta2 * u, np.abs(grad))
    # 파라미터 업데이트
    pt = pt - (lr_adamax / (u + 1e-8)) * m
    path_adamax.append(pt)
path_adamax = np.array(path_adamax)

# ==========================================
# 3. 시각화 (Explore & Discover)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# [Plot 1] 에폭(Epoch)과 학습률(Learning Rate)
axes[0].plot(epochs, loss_crawling, label='LR=0.001 (Too Small)', color='blue', lw=3)
axes[0].plot(epochs, loss_optimal, label='LR=0.01 (Optimal)', color='green', lw=3)
axes[0].plot(epochs, loss_exploding, label='LR=10.0 (Too Big)', color='red', lw=3)
axes[0].set_ylim([0, 5])
axes[0].set_title('1. Cross Entropy Loss vs Epochs\n[The Art of Stride]', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epochs (Number of Passes)')
axes[0].set_ylabel('Loss (Error)')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# 손실 지형 등고선 배경 세팅
X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-4, 4, 100))
Z = loss_function(X, Y)

# [Plot 2] 기초 옵티마이저: SGD
axes[1].contour(X, Y, Z, levels=np.logspace(-0.5, 2.5, 20), cmap='gray', alpha=0.3)
axes[1].plot(path_sgd[:, 0], path_sgd[:, 1], 'ro-', lw=2, markersize=6, label='SGD Path')
axes[1].plot(0, 0, 'y*', markersize=20, markeredgecolor='black', label='Global Minimum')
axes[1].set_title('2. SGD (Stochastic Gradient Descent)\n[Zig-zagging in the valley]', fontsize=14, fontweight='bold', color='red')
axes[1].set_xlim([-5, 5])
axes[1].set_ylim([-4, 4])
axes[1].legend(loc='lower right')

# [Plot 3] 고급 옵티마이저: Adamax
axes[2].contour(X, Y, Z, levels=np.logspace(-0.5, 2.5, 20), cmap='gray', alpha=0.3)
axes[2].plot(path_adamax[:, 0], path_adamax[:, 1], 'bo-', lw=2, markersize=6, label='Adamax Path')
axes[2].plot(0, 0, 'y*', markersize=20, markeredgecolor='black', label='Global Minimum')
axes[2].set_title('3. Advanced Optimizer: Adamax\n[Smooth & Adaptive Step]', fontsize=14, fontweight='bold', color='blue')
axes[2].set_xlim([-5, 5])
axes[2].set_ylim([-4, 4])
axes[2].legend(loc='lower right')

plt.tight_layout()
plt.show()

# ==========================================
# 4. 터미널 출력 (강의 진행 시나리오)
# ==========================================
print("\n" + "="*75)
print("🏃‍♂️ [탐구 미션] 훈련의 기술: 채찍과 당근, 그리고 보폭의 예술")
print("="*75)
print("Q1. 모델이 예측을 틀렸을 때 주어지는 벌점(Cross Entropy 오차)을 줄여야 합니다.")
print("    [Plot 1]에서 최적의 지점을 향해 걸어갈 때 보폭(Learning Rate)을 10.0으로 너무 크게 잡으면?")
print("    -> (관찰) 빨간 선처럼 계곡을 휙휙 건너뛰다 못해 우주로 발산(Exploding)해 버립니다!")
print("    -> (해결) 에폭(Epoch)을 충분히 주면서(초록 선) 적절한 보폭으로 내려가야 합니다.")
print("-" * 75)
print("Q2. 산꼭대기에서 안대를 쓰고 가장 낮은 골짜기(Global Minimum 🌟)를 찾아 내려가야 합니다.")
print("    [Plot 2]의 SGD(기본 경사 하강법)는 가파른 절벽 쪽으로만 크게 움직이는 경향이 있습니다.")
print("    -> (관찰) 빨간 점들이 좌우로 심하게 요동치며(Zig-zag) 비효율적으로 내려갑니다.")
print("-" * 75)
print("Q3. 그렇다면 어떻게 개선할 수 있을까요? (베이스라인 코드의 개선 전략)")
print("    -> (해결) 과거에 내려가던 '관성(Momentum)'을 기억하고, 보폭을 스스로 조절하는 똑똑한 내비게이션,")
print("       [Plot 3]의 'Adamax(Adam 계열)' 옵티마이저로 교체합니다.")
print("    -> (관찰) 파란 점들이 계곡의 중앙을 타고 미끄러지듯 빠르고 부드럽게 목표점에 도달합니다!")
print("="*75 + "\n")
