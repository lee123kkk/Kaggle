#DropOut_BathNormalization

'''
[중급 1] 학습 모드 vs 평가 모드 (Train vs Eval)

문제: PyTorch 모델을 학습할 때 model.train()을 호출하고, 평가할 때 model.eval()을 호출합니다.

이 두 모드의 전환이 이미지 속 '드롭아웃(Dropout)'​과 '배치 정규화(Batch Normalization)'에 구체적으로 어떤 영향을 미치나요?

'''


'''
드롭 아웃은 학습 모드에서는 설정된 확률에 따라 무작위로 노드를 꺼내버린다. 
살아남은 노드들은 값이 꺼진 노드들의 몫까지 보상하기 위해 커진다. 
평가모드에서는 드롭아웃이 완전히 꺼진다. 

배치 정규화는 학습모드에서는 현재 들어온 데이터의 평균과 분산을 계산하여 정규화한다. 
그리고 이동 평균과 이동 분산을 업데이트 한다. 
평가모드에서는 학습 과정에서 미리 계산해 둔 이동 평균과 이동 분산을 가져와서 고정된 상태에서 정규화한다.
'''

import torch
import torch.nn as nn

# 결과를 보기 쉽게 소수점 둘째 자리까지 출력하도록 설정
torch.set_printoptions(precision=2, sci_mode=False)

# ==========================================
# 1. 드롭아웃 (Dropout) 동작 비교
# ==========================================
print("=== [1] Dropout (p=0.5) 동작 비교 ===")
dropout = nn.Dropout(p=0.5)

# 값이 1로 가득 찬 텐서 (크기: 10)
input_tensor = torch.ones(10)

# [학습 모드]
dropout.train()
train_out = dropout(input_tensor)
print(f"Train 모드 출력:\n{train_out}")
print("-> 설명: 절반(50%)이 0으로 꺼지고, 살아남은 값은 보상을 위해 2배(1/0.5)가 됨!\n")

# [평가 모드]
dropout.eval()
eval_out = dropout(input_tensor)
print(f"Eval 모드 출력:\n{eval_out}")
print("-> 설명: 무작위성이 사라지고, 모든 값이 원본 그대로(1.0) 통과함!\n")


# ==========================================
# 2. 배치 정규화 (Batch Normalization) 동작 비교
# ==========================================
print("=== [2] Batch Normalization 동작 비교 ===")
# 1차원 특성(Feature) 1개를 갖는 BatchNorm 생성
batch_norm = nn.BatchNorm1d(num_features=1)

# 테스트용 미니 배치 데이터 (배치 크기 3, 특성 1)
# 값: 10.0, 20.0, 30.0 (평균: 20.0, 분산 통계치 존재)
batch_data = torch.tensor([[10.0], [20.0], [30.0]])

# [학습 모드]
batch_norm.train()
train_bn_out = batch_norm(batch_data)
print(f"입력 데이터:\n{batch_data.view(-1)}")
print(f"Train 모드 정규화 출력:\n{train_bn_out.view(-1)}")
print("-> 설명: 현재 3개 값의 평균(20)을 기준으로 즉석에서 정규화됨 (결과: -1.22, 0, 1.22)\n")

# [평가 모드]
batch_norm.eval()
# 평가 모드에서는 '현재 데이터'가 아니라, 학습 중 누적된 '이동 평균(Running Mean)'을 사용합니다.
# 방금 1번의 학습(train)을 거치며 이동 평균이 0에서 2.0으로 미세하게 변했습니다.
eval_bn_out = batch_norm(batch_data)
print(f"Eval 모드 정규화 출력:\n{eval_bn_out.view(-1)}")
print(f"(참고) 저장된 이동 평균(Running Mean): {batch_norm.running_mean.item():.2f}")
print("-> 설명: 현재 배치의 평균(20)을 완전히 무시하고, 저장된 이동 평균(2.0)을 기준으로 정규화하여 출력값이 완전히 다름!\n")
