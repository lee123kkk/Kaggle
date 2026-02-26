#5_Box_Plot

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd

# [상황 설정]
# 정답인 경우: 확률이 높음 (0.8 ~ 1.0)
# 오답인 경우: 확률이 낮거나 애매함 (0.3 ~ 0.6) -> 모델이 헷갈려 함 (이상적)
# 오답인데 확률이 높음 (0.9) -> 모델이 "확신을 가지고 틀림" (위험 신호)

n_samples = 500
correct_probs = np.random.beta(5, 1, n_samples)  # 1에 가까운 분포
wrong_probs = np.random.beta(2, 2, n_samples)    # 0.5 근처 분포

df = pd.DataFrame({
    'Confidence': np.concatenate([correct_probs, wrong_probs]),
    'Result': ['Correct'] * n_samples + ['Wrong'] * n_samples
})

# 박스 플롯 시각화
plt.figure(figsize=(8, 6))
sns.boxplot(x='Result', y='Confidence', data=df, palette='Set2')
plt.title('Prediction Confidence Distribution')
plt.ylabel('Softmax Probability (Max)')
plt.grid(True, axis='y', alpha=0.3)
plt.show()
#=============================================================
# 박스 플롯
# 모델이 정답을 맞췄을 때와 틀렸을 때 얼마나 확신을 가지고 있었는지 분석한다.
# wrong 모델이 위쪽에 있으면 과잉 확신이다.
# 라벨 스무딩을 활용해서 모델이 90% 정도 확신을 갖게 해서 일반화 성능을 높인다.
