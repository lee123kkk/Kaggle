#Multi_variable_linear_Regression_PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. GPU 디바이스 설정 (앞으로 모든 코드에 기본으로 적용됩니다)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")

# 2. 데이터 준비 및 GPU 할당
x_data = np.array([[73., 80., 75.],
                   [93., 88., 93.],
                   [89., 91., 90.],
                   [96., 98., 100.],
                   [73., 66., 70.]], dtype=np.float32)
y_data = np.array([[152.],
                   [185.],
                   [180.],
                   [196.],
                   [142.]], dtype=np.float32)

# NumPy 배열을 PyTorch 텐서로 변환한 뒤 GPU 메모리로 이동시킵니다.
x_train = torch.tensor(x_data).to(device)
y_train = torch.tensor(y_data).to(device)

# 3. 모델 정의
# Keras의 Dense(units=1, input_dim=3)과 완벽히 동일한 역할입니다.
model = nn.Linear(in_features=3, out_features=1).to(device)

# 4. 손실 함수와 옵티마이저 설정
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-5)

# 5. 모델 학습 (Keras의 model.fit 역할)
epochs = 100
model.train() # 학습 모드 전환

for epoch in range(epochs):
    optimizer.zero_grad()              # 기울기 초기화
    hypothesis = model(x_train)        # 예측값 계산 (순전파)
    loss = criterion(hypothesis, y_train) # 오차(MSE) 계산
    loss.backward()                    # 역전파 (기울기 계산)
    optimizer.step()                   # 가중치 업데이트

# 6. 결과 예측 (Keras의 model.predict 역할)
model.eval() # 평가 모드 전환
# 예측할 데이터 역시 텐서로 변환 후 GPU로 보내야 연산이 가능합니다.
x_test = torch.tensor([[72., 93., 90.]], dtype=torch.float32).to(device)

with torch.no_grad():
    y_predict = model(x_test)

# 출력 시에는 다시 CPU로 가져와서 NumPy 배열로 변환합니다.
print("예측 결과:\n", y_predict.cpu().numpy())
#==========================================================
# 다중 변수 선형 회귀 코드 pytorch 변환

# 다중 변수 입력 형태 정의:
    # Dense -> nn.Linear
    # 예: Dense(units=1, input_dim=3)으로 입력 변수가 3개임을 지정
    #     nn.Linear(in_features=3, out_features=1) (들어오는 특성(Feature)이 3개이고 나가는 출력이 1개)
# 활성화 함수 생략 제거
# 데이터 텐서 변환 및 명시적 GPU 할당
# 추론 시 기울기 계산 방지

