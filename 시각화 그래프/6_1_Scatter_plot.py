# 6_1_Scatter_plot

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.manifold import TSNE

# 1. 가상 특징 벡터(Feature Vector) 데이터 생성
# 상황: 잘 학습된 CNN 모델이 이미지를 통과시켜 마지막 분류기 직전에 50차원의 특징을 뽑아낸 상태를 시뮬레이션합니다.
# 샘플 1000개, 클래스 10개, 각 샘플은 50차원의 벡터값을 가집니다.
X_features, y_labels = make_blobs(n_samples=1000, centers=10, n_features=50, 
                                  cluster_std=3.0, random_state=42)

# 2. t-SNE를 이용한 차원 축소 (50차원 -> 2차원)
# 사람이 눈으로 볼 수 없는 50차원 공간의 거리 정보를 최대한 보존하면서 2차원 평면으로 압축합니다.
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_features)

# 3. 산점도(Scatter Plot) 시각화
plt.figure(figsize=(12, 8))

# seaborn의 scatterplot을 이용해 2차원으로 축소된 좌표를 찍고, 실제 라벨(y_labels)별로 색상을 다르게 칠합니다.
# 'tab10' 팔레트는 10개의 명확히 구분되는 색상을 제공합니다.
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_labels, 
                palette='tab10', s=60, alpha=0.8, legend='full')

# 그래프 꾸미기
plt.title("t-SNE Visualization of CNN Feature Vectors (50D -> 2D)", fontsize=16)
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")

# 범례(Legend) 위치 조정
plt.legend(title="Class (0-9)", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()

#=========================================================
# 산점도 
# 군집의 분리도를 확인할 수 있다.
# 군집이 중첩되어 있다면 추출기가 두 객체의 차이점을 파악하지 못한 것이다.
# 분류기의 성능은 고차원 특징들을 클래스별로 얼마나 잘 뭉쳐져 있는지에 달려 있다.