#1_1_Count_Plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 가상 데이터 생성 (동료가 만든 불균형한 csv 데이터셋 시뮬레이션)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 의도적으로 'cat'과 'dog' 클래스의 데이터를 현저히 적게 만듭니다.
# 정상 클래스는 약 5000개, 부족한 클래스는 약 500개로 설정
simulated_data = []
for cls in class_names:
    if cls in ['cat', 'dog']:
        count = np.random.randint(400, 600)  # 소수 클래스: 400~600개
    else:
        count = np.random.randint(4800, 5200) # 다수 클래스: 4800~5200개
    simulated_data.extend([cls] * count)

# 데이터프레임으로 변환 (마치 pd.read_csv('my_cifar_subset.csv')를 한 것과 동일한 상태)
df = pd.DataFrame({'label': simulated_data})

# 2. 카운트플롯 시각화 (데이터 불균형 탐지)
plt.figure(figsize=(12, 5))

# seaborn.countplot을 사용하면 데이터프레임 내의 범주형 데이터 개수를 자동으로 세어 막대그래프로 그려줍니다.
sns.countplot(data=df, x='label', order=class_names, palette='Set2')

# 그래프 꾸미기
plt.title("Class Distribution of 'my_cifar_subset' (Finding the Imbalance)")
plt.xlabel("Class Name")
plt.ylabel("Number of Images")

# 막대 위에 정확한 개수를 텍스트로 표시 (더 직관적인 분석을 위해 추가)
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 3), 
                textcoords='offset points')

plt.show()
#=====================================================================
# 다른 클래스들은 5000개에 가까운 데이터를 가지고 있지만 cat과 dog 클래스만 500개 정도이다.
# 딥러닝 모델은 데이터가 많은 쪽에 유리하게 학습된다.

# 모델이 특정 클래스에서만 성능이 떨어진다면 클래스 불균형 상태인지 확인해봐야 한다.
