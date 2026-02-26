# Parameter_Counting

'''
[중급 2] 파라미터 수 계산 (Parameter Counting)

문제: CNN이 완전 연결 신경망(Fully Connected Network)보다 이미지 처리에 효율적인 이유 중 하나는 파라미터 공유입니다.

다음 두 레이어의 학습 가능한 파라미터(가중치 + 편향) 개수를 각각 계산하고 비교하세요. (입력 채널은 32라고 가정)

Conv Layer: nn.Conv2d(32, 64, kernel_size=3, padding=1)

FC Layer: 입력 뉴런 32개, 출력 뉴런 64개인 nn.Linear(32, 64) 
(단, CNN과 직접 비교를 위해 커널 크기 등의 공간 정보는 무시하고 단순 채널 간 연결로만 가정)
'''

'''
conv2d는 총 파라미터가 64X32X3X3+64=18496개이다.
FC Layer는 총 파라미터의 개수가 64X32+64=2112개이다.

FC 레이어가 총 파라미터의 수가 적다.
32x32 해상ㅇ도의 이미지를 처리할때 cnn은 이미지가 아무리 커져도 파라미터 수는 여전히 18496개이다.
fc는 이미지를 1차원으로 펴야 하므로 32x32x32=32768이된다. 출력노드와 1:1로 연결하면 약 21억게의 가중치가 필요하다.

cnn은 작은 필터 하나를 이미지 전체에 재사용하기 때문에 효율적이다.
'''


import torch
import torch.nn as nn

# 1. 레이어 정의
conv_layer = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
fc_layer = nn.Linear(in_features=32, out_features=64)

# 2. 파라미터 개수를 세는 헬퍼 함수
def count_parameters(layer):
    # layer.parameters()는 가중치와 편향 텐서를 반환합니다.
    # p.numel()은 해당 텐서안에 있는 요소(element)의 총 개수를 구합니다.
    return sum(p.numel() for p in layer.parameters() if p.requires_grad)

# 3. 계산 및 출력
conv_params = count_parameters(conv_layer)
fc_params = count_parameters(fc_layer)

print(f"=== Parameter Count ===")
print(f"1. Conv2d Layer: {conv_params:,} 개")
print(f"2. Linear Layer:  {fc_params:,} 개")

# (보너스) 실제 텐서의 형태(Shape) 확인
print("\n=== Tensor Shapes ===")
print(f"Conv2d Weights: {conv_layer.weight.shape} -> {conv_layer.weight.numel()}개")
print(f"Conv2d Biases:  {conv_layer.bias.shape} -> {conv_layer.bias.numel()}개")
print(f"Linear Weights: {fc_layer.weight.shape} -> {fc_layer.weight.numel()}개")