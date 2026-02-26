# 3_1_Loss_Accuaracy

import matplotlib.pyplot as plt
import numpy as np

# 가상의 에포크(Epoch) 50번 설정
epochs = np.arange(1, 51)

# [상황 1] 모델 A 가상 데이터: 과적합(Overfitting)
# Train Loss는 0.1에 수렴하도록 지수 감소, Val Loss는 떨어지다가 다시 2.5를 향해 치솟도록 설정
train_loss_A = 2.0 * np.exp(-0.1 * epochs) + 0.1
val_loss_A = 2.0 * np.exp(-0.1 * epochs) + 0.05 * epochs

# [상황 2] 모델 B 가상 데이터: 과소적합(Underfitting) 혹은 학습 정체
# Train과 Val Loss 모두 천천히 떨어지다가 0.5 근처에서 평행선을 달리도록 설정
train_loss_B = 1.0 * np.exp(-0.05 * epochs) + 0.45
val_loss_B = 1.0 * np.exp(-0.05 * epochs) + 0.5

# 시각화 (Line Plot)
plt.figure(figsize=(14, 5))

# 첫 번째 그래프: 모델 A (과적합)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_A, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss_A, label='Validation Loss', color='red', linewidth=2)
plt.title("Model A: Overfitting (The Gap Widens)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# 두 번째 그래프: 모델 B (과소적합)
plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss_B, label='Train Loss', color='blue', linewidth=2)
plt.plot(epochs, val_loss_B, label='Validation Loss', color='red', linewidth=2)
plt.title("Model B: Underfitting (High Bias / Plateau)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
#=========================================================
# 두 선을 바탕으로 모델의 성능을 파악한다
# 모델 A는 훈련 데이터의 정답을 통쨰로 암기해버리는 과적합이 발생했고,
# 모델 B는 파란선과 빨간 선이 붙어있지만, 0.5라는 높은 loss에서 멈춰 있는 과소적합상태이다.

# 과적합에 대처하려먼 dropout등의 방안이 있고, 과소적합을 해결하려면 은닉층의 개수를 늘리는 등의 방안이 있다.
