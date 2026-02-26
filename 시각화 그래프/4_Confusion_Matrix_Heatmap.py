#4_Confusion_Matrix_Heatmap

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# [상황 설정]
# CIFAR-10 클래스: 0:비행기, 1:자동차, 2:새, 3:고양이, 4:사슴, 5:개, ...
# 모델이 '고양이(3)'를 '개(5)'로 자주 착각하는 상황을 가정한 더미 데이터
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# 실제값(y_true)과 예측값(y_pred) 생성 (가상)
y_true = np.random.randint(0, 10, 1000)
y_pred = y_true.copy()

# 고양이(3) 중 30%를 개(5)로 오분류하도록 조작
mask_cat = (y_true == 3)
num_confused = int(np.sum(mask_cat) * 0.3)
idxs = np.where(mask_cat)[0][:num_confused]
y_pred[idxs] = 5  # 고양이를 개로 예측

# 1. 혼동 행렬 계산
cm = confusion_matrix(y_true, y_pred)

# 2. 히트맵 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Cat vs Dog Confusion)')
plt.show()

#=================================================================
# 혼돈 행렬 히트맵
# 모델이 가장 헷갈려 하는 클래스를 보여준다.

# 이 모델에서는 개와 고양이를 헷갈려 하므로 Mixup같은 증강 기법을 사용해야 한다.
