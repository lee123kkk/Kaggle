# Residual_Connection

import torch
import torch.nn as nn

# ==========================================
# 1. ResBlock 클래스 정의
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        
        # 요구사항: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm
        # (BatchNorm을 쓸 때는 bias=False로 설정하는 것이 메모리 효율에 좋습니다)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # 1. 원본 입력 x를 따로 기억해둡니다. (이것이 고속도로를 탈 데이터입니다!)
        shortcut = x 
        
        # 2. 메인 경로 (Main Path) 연산 수행
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 3. [핵심] 잔차 연결 (Skip Connection)
        # 메인 경로를 통과한 결과(out)에, 아까 기억해둔 원본 입력(shortcut)을 그대로 더해줍니다.
        out += shortcut 
        
        # 4. 더해진 결과에 최종 활성화 함수를 적용합니다.
        out = self.relu(out)
        
        return out

# ==========================================
# 2. 동작 검증 (더미 데이터 테스트)
# ==========================================
# 가상의 GPU 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 배치 사이즈 1, 채널 64, 해상도 32x32 인 더미 이미지 텐서 생성
x = torch.randn(1, 64, 32, 32).to(device)

# 입력과 동일한 채널(64)을 유지하는 ResBlock 인스턴스 생성
res_block = ResBlock(channels=64).to(device)

# 이미지를 블록에 통과시킵니다.
output = res_block(x)

print(f"입력 텐서 크기: {x.shape}")
print(f"출력 텐서 크기: {output.shape}")
print("\n[성공] 입력과 동일한 크기의 텐서가 에러 없이 출력되었습니다! 잔차 연결이 정상 작동합니다.")
#======================================================================
# 입력값에 출력을 단순히 더채주는 것만으로 기울기 소실 없이 안정적이 학습이 가능해진다.
#  
#