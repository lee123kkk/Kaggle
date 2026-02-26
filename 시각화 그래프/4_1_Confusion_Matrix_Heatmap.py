#4_1_Confusion_Matrix_Heatmap

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# CIFAR-10 클래스 이름 정의
classes = ['airplane', 'auto', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. 가상 예측 결과 시뮬레이션 (각 클래스당 1,000개의 테스트 데이터 가정)
y_true = []
y_pred = []

# 랜덤 시드 고정 (재현성)
np.random.seed(42)

for i in range(10):
    y_true.extend([i] * 1000)
    
    # [상황 부여] 실제 라벨이 트럭(9)인 경우: 심각한 혼동 발생
    if i == 9: 
        # 500개만 맞추고, 무려 400개를 자동차(1)로 착각! (나머지 100개는 랜덤 오답)
        preds = [9] * 500 + [1] * 400 + list(np.random.randint(0, 9, 100))
        y_pred.extend(preds)
    # 나머지 클래스들: 비교적 평범하게 850개 정도 맞춘다고 가정
    else:
        preds = [i] * 850 + list(np.random.randint(0, 10, 150))
        y_pred.extend(preds)

# 2. 혼동 행렬(Confusion Matrix) 계산
# 행(Row)은 실제 정답(True), 열(Column)은 모델의 예측(Predicted)을 의미합니다.
cm = confusion_matrix(y_true, y_pred)

# 3. Heatmap 시각화
plt.figure(figsize=(10, 8))
# annot=True: 셀 안에 숫자 표시 / fmt='d': 정수형 표기 / cmap='Reds': 붉은색 테마 (경고/에러 강조 느낌)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=classes, yticklabels=classes)

# 그래프 꾸미기
plt.title("CIFAR-10 Confusion Matrix: Why is the model failing on Trucks?")
plt.xlabel("Predicted Label (What the model thought)")
plt.ylabel("True Label (The actual answer)")
plt.show()
#=========================================================
# 트럭의 결과를 확인해보면  turck으로 예측한 것은 500개 정도이고 auto라고 예측한 것이 400개나 된다.
# 트럭과 자동차는 모두 바퀴가 4개이상이고 도로 위를 달리는 등 시각적으로 공통된 특징이 많기 때문에 생기는 문제이다.

# 정확도만 확인하지 말고 세부적인 약점 파악을 위해서 혼동 행렬을 통해서 교차검증을 해봐야 한다.
