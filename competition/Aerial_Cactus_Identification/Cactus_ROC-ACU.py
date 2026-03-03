# Cactus_ROC-ACU

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

# ==========================================
# 1. 캑터스 대회 데이터 시뮬레이션
# (선인장 있음: 75%, 선인장 없음: 25%)
# ==========================================
np.random.seed(42)
n_samples = 1000
y_true = np.array([1] * 750 + [0] * 250) # 실제 정답
np.random.shuffle(y_true)

# ==========================================
# 2. 두 가지 가짜 모델(AI) 생성
# ==========================================
# 🤖 [모델 A] 바보 모델 (찍기 신공)
# "분석하기 귀찮아. 그냥 다 선인장이 있다고 우기자!"
# (확률을 0.51~0.99 사이로 랜덤하게 내뱉음 -> 임계값 0.5 기준 무조건 '1'로 예측)
y_scores_dumb = np.random.uniform(0.51, 0.99, n_samples)
y_pred_dumb = (y_scores_dumb >= 0.5).astype(int)

# 🧠 [모델 B] 똑똑한 모델 (진짜 실력자)
# "어느 정도 패턴을 찾았어. 선인장을 구별할 수 있어!"
y_scores_smart = y_true * np.random.uniform(0.6, 1.0, n_samples) + (1 - y_true) * np.random.uniform(0.0, 0.4, n_samples)
# 약간의 실수(노이즈) 섞어주기
noise_indices = np.random.choice(n_samples, 100, replace=False)
y_scores_smart[noise_indices] = 1.0 - y_scores_smart[noise_indices]
y_pred_smart = (y_scores_smart >= 0.5).astype(int)

# ==========================================
# 3. 평가 지표 계산
# ==========================================
acc_dumb = accuracy_score(y_true, y_pred_dumb)
acc_smart = accuracy_score(y_true, y_pred_smart)

fpr_dumb, tpr_dumb, _ = roc_curve(y_true, y_scores_dumb)
auc_dumb = auc(fpr_dumb, tpr_dumb)

fpr_smart, tpr_smart, _ = roc_curve(y_true, y_scores_smart)
auc_smart = auc(fpr_smart, tpr_smart)

cm_dumb = confusion_matrix(y_true, y_pred_dumb)
cm_smart = confusion_matrix(y_true, y_pred_smart)

# ==========================================
# 4. 터미널 출력 (질문과 관찰 유도)
# ==========================================
print("\n" + "="*60)
print("🌵 [탐구 미션] 정확도(Accuracy)의 함정을 찾아라!")
print("="*60)
print(f"전체 데이터 {n_samples}장 중, 선인장 사진은 무려 75%인 {sum(y_true)}장입니다.\n")

print(f"🤖 [모델 A] 바보 모델 (무조건 '선인장 있음'으로 찍음)")
print(f" - 정확도(Accuracy): {acc_dumb*100:.1f}%  <-- (질문: 오! 학습을 안 했는데 75점이나? 좋은 모델일까요?)")
print(f" - ROC AUC 점수    : {auc_dumb:.3f}      <-- (어라? AUC는 왜 바닥이죠?)")
print(f" - 혼동 행렬:\n{cm_dumb}\n")

print(f"🧠 [모델 B] 똑똑한 모델 (진짜 선인장을 구분함)")
print(f" - 정확도(Accuracy): {acc_smart*100:.1f}%")
print(f" - ROC AUC 점수    : {auc_smart:.3f}      <-- (이것이 이 모델의 진짜 실력입니다!)")
print(f" - 혼동 행렬:\n{cm_smart}\n")
print("="*60 + "\n")

# ==========================================
# 5. 시각화 (Discover & Aha-moment)
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# [Plot 1] 바보 모델의 Confusion Matrix
sns.heatmap(cm_dumb, annot=True, fmt='d', cmap='Reds', ax=axes[0], cbar=False, annot_kws={"size": 16})
axes[0].set_title(f'Model A (Dumb) CM\nAccuracy: {acc_dumb*100:.1f}%', fontsize=14, fontweight='bold', color='red')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

# [Plot 2] 똑똑한 모델의 Confusion Matrix
sns.heatmap(cm_smart, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False, annot_kws={"size": 16})
axes[1].set_title(f'Model B (Smart) CM\nAccuracy: {acc_smart*100:.1f}%', fontsize=14, fontweight='bold', color='blue')
axes[1].set_xlabel('Predicted Label')
axes[1].set_ylabel('True Label')

# [Plot 3] ROC Curve 비교
axes[2].plot(fpr_dumb, tpr_dumb, color='red', lw=3, linestyle=':', label=f'Model A (AUC = {auc_dumb:.3f})')
axes[2].plot(fpr_smart, tpr_smart, color='blue', lw=3, label=f'Model B (AUC = {auc_smart:.3f})')
axes[2].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('False Positive Rate (FPR)')
axes[2].set_ylabel('True Positive Rate (TPR)')
axes[2].set_title('ROC Curve Comparison\n(Why Accuracy is a Lie)', fontsize=14, fontweight='bold')
axes[2].legend(loc="lower right", fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
