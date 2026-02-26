# 6_Scatter_Plot

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs

# [상황 설정]
# 모델의 마지막 레이어(FC Layer) 직전에서 나온 64차원 특징 벡터라고 가정
# 10개 클래스가 뭉쳐 있긴 하지만, 일부는 겹쳐 있는 상황
X_features, y_labels = make_blobs(n_samples=1000, centers=10, n_features=64, 
                                  cluster_std=2.0, random_state=42)

# 1. t-SNE로 2차원 축소 (시간이 좀 걸릴 수 있음)
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_embedded = tsne.fit_transform(X_features)

# 2. 산점도 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_labels, cmap='jet', alpha=0.6)
plt.colorbar(scatter, label='Class Label')
plt.title('t-SNE Visualization of Feature Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()

#==================================================================
# 각 클래스가 공간상에서 잘 뭉쳐져 있는지 확인한다.
# 같은 클래스 점들이 뭉쳐 있으면 좋다. 다른 클래스가 뭉쳐 있으면 성능 개선을 해야 한다.
# 거리 기반 손실 함수 등을 통해서 개선할 수 있다.

