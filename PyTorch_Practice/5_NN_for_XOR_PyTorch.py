# 5_NN_for_XOR_PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

x_train = torch.tensor(x_data).to(device)
y_train = torch.tensor(y_data).to(device)

# [튜닝 포인트 1] 은닉층의 노드(units) 수를 2개에서 10개로 늘려 모델의 표현력을 높였습니다.
model = nn.Sequential(
    nn.Linear(2, 10),  
    nn.Sigmoid(),      
    nn.Linear(10, 1),  
    nn.Sigmoid()       
).to(device)

criterion = nn.BCELoss()
# [튜닝 포인트 2] SGD 대신 훨씬 빠르고 안정적으로 수렴하는 Adam 옵티마이저로 변경했습니다.
# 학습률(lr)은 Adam에 맞게 0.01로 조정했습니다.
optimizer = optim.Adam(model.parameters(), lr=0.01)

# [튜닝 포인트 3] 에포크 수를 조금 더 넉넉하게 주어 확실히 수렴하도록 합니다.
epochs = 3000
model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(x_train)
    print('Prediction: \n', predictions.cpu().numpy())
    
    predicted_classes = (predictions >= 0.5).float()
    correct = (predicted_classes == y_train).sum().item()
    accuracy = correct / len(y_train)
    print('Accuracy: ', accuracy)

#=============================================================
# XOR문제 해결을 위한 다층 퍼셉트론 구조 Pytorch

# keras에서는 model.add()를 4번 연속해서 호출해서 층을 깊게 쌓았지만, 
# pytorch에서는 nn.Sequential안에 차례대로 나열해서 구현해야 한다.

# Keras에서는 input_dim만 지정하면 다음 층의 크기를 자동으로 추론하지만, 
# Pytorch에서는 앞 층의 출력과 다음 층의 입력이 정확히 일치하도록 적어줘야 한다.

# keras의 tf.model.evaluate()는 손실값과 설정한 메트릭(정확도 등)을 배열 형태로 즉시 반환해주지만, 
# PyTorch는 별도의 내장 evaluate 함수가 없으므로 학습 루프 밖에서 model.eval() 상태로 직접 정확도를 계산하여 구현했다.

#=============================================================
# 모델이 지역 최적점에 빠져 학습을 제대로 하지 못했다.
# 은닉층의 노드 수와 옵티마이저를 튜닝해야 한다.

# 은닉층의 크기를 nn.Linear(2,2)에서 nn.Linear(2,10)로 바꿔서 복잡한 패턴도 파악할 수 있게 했다.
# 옵티마이저를  SGD에서 Adam을 변경해서 성능을 강화시켰다.




