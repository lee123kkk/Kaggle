# Demension_Calculation

'''
문제: CIFAR-10 이미지는 32 x 32 크기(RGB 3채널)입니다.

다음 Conv2d 레이어를 통과했을 때 출력되는 특성 맵(Feature Map)의 크기(Height, Width)는 얼마입니까? 
(공식을 사용하여 계산 과정을 서술하세요.)

nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=2)

'''

'''
O=(I-K+2P) / s + 1
O: 출력 크기 
I: 입력 크기 = 32
K: 커널 크기 = 3
P: 패딩 = 1
S: 스트라이드 = 2

출력되는 특성 맵의 공간적 크기는 16X16이 된다.

'''

# 검증용 Pytorch 코드

import torch
import torch.nn as nn

# GPU 설정 (우리의 약속!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 가상의 CIFAR-10 이미지 텐서 생성
# PyTorch 입력 형태: (Batch Size, Channels, Height, Width)
# 임의로 배치 사이즈 1, 3채널(RGB), 32x32 크기의 이미지를 생성하여 GPU로 보냅니다.
dummy_input = torch.randn(1, 3, 32, 32).to(device)

print(f"입력 텐서 크기: {dummy_input.shape}")

# 2. 문제에 제시된 Conv2d 레이어 정의 및 GPU 할당
conv_layer = nn.Conv2d(in_channels=3, 
                       out_channels=16, 
                       kernel_size=3, 
                       padding=1, 
                       stride=2).to(device)

# 3. 이미지를 레이어에 통과시키기
output = conv_layer(dummy_input)

print(f"출력 텐서 크기: {output.shape}")