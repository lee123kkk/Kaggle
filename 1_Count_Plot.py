import matplotlib.pyplot as plt
import seaborn as sns
import torchvision
import numpy as np

# 1. CIFAR-10 데이터 로드 (Train set)
# root 파라미터로 다운로드 위치를 지정하고, train=True로 학습용 데이터를 가져옵니다.
train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 2. 라벨(정답) 및 클래스 이름 추출
# targets에는 0~9까지의 정답 인덱스가, classes에는 'airplane', 'automobile' 등의 실제 이름이 들어있습니다.
labels = train_ds.targets
class_names = train_ds.classes

# 3. 카운트플롯 시각화 (데이터 분포 확인)
plt.figure(figsize=(12, 5))
# seaborn을 이용해 막대그래프로 각 라벨의 개수를 셉니다. palette는 색상 테마입니다.
sns.countplot(x=labels, palette='viridis')

# 그래프 꾸미기
plt.title("CIFAR-10 Class Distribution")
plt.xlabel("Class Index")
# x축의 숫자(0~9) 대신 알아보기 쉽게 실제 클래스 이름으로 바꿔 달아줍니다.
plt.xticks(ticks=range(10), labels=class_names)
plt.ylabel("Number of Images")

# 그래프 출력
plt.show()
#===============================================================
# 카운트 플롯
# 각 클래스별로 데이터의 크기가 어떤지 확인한다.

# 10개의 클래스에 각가 5000개씩의 데이터가 있으므로 균형잡혀 있다.
# 복잡한 전처리가 필요없고, 정확도로 모델의 성능을 판단할 수 있다.
