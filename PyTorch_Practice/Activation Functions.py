# Activation Functions

'''
[초급 2] 활성화 함수의 선택 (Activation Functions)

문제: 이미지 분류 모델의 은닉층(Hidden Layer)에는 주로 ReLU를 사용하고, 마지막 출력층에는 Softmax(다중 분류)를 사용합니다.

왜 은닉층에 Sigmoid 대신 ReLU를 선호하나요? (핵심 키워드: 기울기 소실)

ReLU의 단점인 "Dying ReLU" 현상을 해결하기 위해 이미지 속 다이어그램에 언급된 어떤 함수를 사용할 수 있나요?

'''


'''
sigmoid함수는 입력값이 아무리 크거나 작아도 0과 1사이의 값으로 짓눌러 버린다.
이 때문에 양 끝으로 갈수록 함수의 기울기가 0에 가까워진다.
층이 깊은 신경망에서 역전파를 할때 0에 가까운 기울기들이 계속 고배지면 기울기가 전달이 되지 않아서 학습이 멈춘다.

ReLU는 양수 입력에 대해 항상 기울기가 1로 일정하다.

Dying ReLU를 현상을 해결하기 위해서 Leaky ReLU를 사용한다. 
음수는 0이 아니라 아주 미세한 기울기를 가지고 아래로 뻗어나간다.

'''


import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# GPU 강제 할당 규칙 적용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 입력 데이터 생성: -5부터 5까지 100개의 점을 만들고 GPU로 보냅니다.
x = torch.linspace(-5, 5, 100).to(device)

# 2. 활성화 함수 적용 (GPU 연산 수행)
y_sigmoid = torch.sigmoid(x)
y_relu = torch.relu(x)
# Leaky ReLU는 음수 영역의 기울기(negative slope)를 설정할 수 있습니다. 
# 시각적 확인을 위해 눈에 띄게 0.1로 설정합니다.
y_leaky_relu = F.leaky_relu(x, negative_slope=0.1)

# 3. 시각화를 위해 CPU 메모리로 가져온 후 NumPy 배열로 변환합니다.
x_np = x.cpu().numpy()
y_sigmoid_np = y_sigmoid.cpu().numpy()
y_relu_np = y_relu.cpu().numpy()
y_leaky_relu_np = y_leaky_relu.cpu().numpy()

# 4. 시각화 (Line Plot)
plt.figure(figsize=(15, 5))

# [그래프 1] Sigmoid
plt.subplot(1, 3, 1)
plt.plot(x_np, y_sigmoid_np, color='blue', linewidth=2)
plt.title("Sigmoid\n(Risk: Vanishing Gradient)")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)

# [그래프 2] ReLU
plt.subplot(1, 3, 2)
plt.plot(x_np, y_relu_np, color='red', linewidth=2)
plt.title("ReLU\n(Risk: Dying ReLU)")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)

# [그래프 3] Leaky ReLU
plt.subplot(1, 3, 3)
plt.plot(x_np, y_leaky_relu_np, color='green', linewidth=2)
plt.title("Leaky ReLU\n(Solution)")
plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()