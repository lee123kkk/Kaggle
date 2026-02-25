# 3_Binary_Classification_PyTorch

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 준비 및 GPU 할당
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.tensor(x_data, dtype=torch.float32).to(device)
y_train = torch.tensor(y_data, dtype=torch.float32).to(device)

# 모델 정의 (Sequential을 이용해 레이어를 묶음)
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
).to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5000
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    
    # 마지막 에포크에서 정확도(Accuracy) 계산
    if epoch == epochs - 1:
        # Sigmoid 결과값이 0.5 이상이면 1, 아니면 0으로 예측
        prediction = (hypothesis >= 0.5).float()
        # 실제 값과 일치하는 개수를 세어 정확도 계산
        correct = (prediction == y_train).sum().item()
        accuracy = correct / len(y_train)

print(f"Accuracy: {accuracy}")

#===============================================================
# 이진 분류 코드 PyTorch로 변환

# sequential과 sigmoid의 결합
# Keras에서 sequential 안에 dense와 activation을 넣었던 방식을 pytorch에서는 nn.Sequential을 사용하여 구현했다.

# 손실 함수: binary_crossentropy ➡️ nn.BCELoss()

# 정확도 수동 계산: keras에서는 compile에서 metrics=['accuracy']만 설정하면 알아서 계산해주지만,
# Pytorch는 직접 구해야 한다. 


