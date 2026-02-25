# 4_Softmax_Regression_Pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_raw = [[1, 2, 1, 1], [2, 1, 3, 2], [3, 1, 3, 4], [4, 1, 5, 5],
         [1, 7, 5, 5], [1, 2, 5, 6], [1, 6, 6, 6], [1, 7, 7, 7]]
y_raw = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 1, 0],
         [0, 1, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]]

x_train = torch.tensor(x_raw, dtype=torch.float32).to(device)
# PyTorch의 CrossEntropyLoss에 맞게 원-핫 인코딩을 클래스 인덱스(0, 1, 2)로 변환합니다.
y_train = torch.tensor(y_raw, dtype=torch.float32).argmax(dim=1).to(device)

nb_classes = 3

# 모델 정의
model = nn.Linear(4, nb_classes).to(device)

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 2000
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_train) # Logits (Softmax 통과 전)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

print('--------------')
# Testing & One-hot encoding
model.eval()

with torch.no_grad():
    # 반복되는 예측 코드를 깔끔하게 함수로 묶었습니다.
    def predict(x_test_data):
        x_test = torch.tensor(x_test_data, dtype=torch.float32).to(device)
        logits = model(x_test)
        probs = torch.softmax(logits, dim=1) # 확률값으로 변환
        predicted_class = torch.argmax(probs, dim=1) # 가장 높은 확률의 클래스 인덱스 추출
        return probs.cpu().numpy(), predicted_class.cpu().numpy()

    a_prob, a_class = predict([[1, 11, 7, 9]])
    print(a_prob, a_class)
    print('--------------')
    
    b_prob, b_class = predict([[1, 3, 4, 3]])
    print(b_prob, b_class)
    print('--------------')
    
    c_prob, c_class = predict([[1, 1, 0, 1]])
    print(c_prob, c_class)
    print('--------------')
    
    all_prob, all_class = predict([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])
    print(all_prob, all_class)

    #==================================================
    # 다중 클래스 분류 코드 PyTorch 변환

    # keras에서는 Dense 레이어 뒤에 Activation('softmax')를 명시적으로 추가했지만
    # pytorch의 nn.CrossEntropyLoss()는 내부적으로 softmax와 loss 계산을 한번에 처리한다.
    # 따라서 모델에는 nn.Linear만 남겨두고, 평가할때만 사용자가 직접 torch.softmax()를 씌워 확률값을 확인한다.

    # keras의 categorical_crossentropy는 원-핫 인코딩 형태의 정답을 그대로 받지만, 
    # PyTorch는 정답 클래스의 인덱스 번호를 1차원 배열 형태로 받는다.

    # Argmax연산을 Pytorch에서는 torch.argmax(dim=1)하나로 처리한다.
