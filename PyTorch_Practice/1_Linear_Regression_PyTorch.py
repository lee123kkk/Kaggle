import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. GPU 디바이스 설정 (가장 중요!)
# CUDA가 사용 가능하면 GPU를, 그렇지 않으면 CPU를 사용하도록 설정합니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 디바이스: {device}")

# 2. 데이터 준비
# PyTorch의 nn.Linear는 (배치 크기, 입력 특성 수) 형태의 2차원 텐서를 요구합니다.
# 따라서 view(-1, 1)을 사용하여 [4] 형태를 [4, 1]로 바꿔줍니다.
# 마지막으로 .to(device)를 붙여 데이터를 GPU로 보냅니다.
x_train = torch.tensor([1, 2, 3, 4], dtype=torch.float32).view(-1, 1).to(device)
y_train = torch.tensor([0, -1, -2, -3], dtype=torch.float32).view(-1, 1).to(device)

# 3. 모델 정의
# Keras의 Dense(units=1, input_dim=1)과 동일한 역할입니다.
# 모델 구조 또한 .to(device)를 통해 GPU 메모리에 올려야 합니다.
model = nn.Linear(in_features=1, out_features=1).to(device)

# 4. 옵티마이저와 손실 함수(MSE) 설정
optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.MSELoss()

# 5. 모델 학습 (Keras의 model.fit 역할)
epochs = 200
model.train() # 모델을 학습 모드로 설정 (Dropout, BatchNorm 등이 있을 때 필수)

for epoch in range(epochs):
    optimizer.zero_grad()              # 1. 이전 루프의 기울기 초기화
    output = model(x_train)            # 2. 예측값 계산 (순전파)
    loss = criterion(output, y_train)  # 3. 오차(Loss) 계산
    loss.backward()                    # 4. 오차 역전파 (기울기 계산)
    optimizer.step()                   # 5. 가중치 업데이트

# 6. 결과 예측 (Keras의 model.predict 역할)
model.eval() # 모델을 평가(추론) 모드로 설정
# 예측할 데이터 역시 형태를 맞추고 GPU로 보내야 합니다.
x_test = torch.tensor([5, 4], dtype=torch.float32).view(-1, 1).to(device)

# 평가 시에는 기울기(Gradient)를 계산할 필요가 없으므로 메모리를 절약합니다.
with torch.no_grad():
    y_predict = model(x_test)

# 출력할 때는 다시 CPU로 가져와서 보기 좋게 numpy 배열로 변환합니다.
print("예측 결과:\n", y_predict.cpu().numpy())

#================================================================
# Tenserflow기반 선형 회귀 코드를 Pythorch로 변환

# PyTorch는 Keras의 .fit()처럼 숨겨진 함수 대신 
# 학습 루프(Training Loop)를 직접 작성해야 한다.

# keras는 1차원 배열도 내부적으로 알아서 처리하지만, PyTorch는 입력 형태에 엄격하여 
# .view(-1, 1)로 차원을 명확히 지정해 주어야 한다.

# 학습루프: zero_grad() -> forward(예측) -> loss -> backward() -> step() 
# 이 5단계가 PyTorch 모델 훈련의 기본 공식

