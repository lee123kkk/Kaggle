# Cactus_Data_Augmentation

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
# 실제 PyTorch 환경 강의 시 추가: import torchvision.transforms as transforms

# ==========================================
# 1. 항공 사진(선인장) 가상 이미지 생성
# ==========================================
img_size = 150
# 사막의 모래 배경 (Desert sand)
img_arr = np.full((img_size, img_size, 3), [237, 201, 175], dtype=np.uint8)
img = Image.fromarray(img_arr)
draw = ImageDraw.Draw(img)

# 위에서 내려다본 선인장 모양 (비대칭적 구조로 그려서 회전/반전 효과가 뚜렷하게 보이게 함)
draw.ellipse([60, 30, 90, 120], fill=(34, 139, 34))   # 몸통
draw.ellipse([30, 60, 65, 80], fill=(34, 139, 34))    # 왼쪽 가지 1
draw.ellipse([30, 40, 45, 70], fill=(34, 139, 34))    # 왼쪽 가지 2
draw.ellipse([85, 45, 115, 65], fill=(34, 139, 34))   # 오른쪽 가지 1
draw.ellipse([100, 30, 115, 55], fill=(34, 139, 34))  # 오른쪽 가지 2

# ==========================================
# 2. 데이터 증강(Data Augmentation) 파이프라인 시뮬레이션
# ==========================================
# 2-1. 단일 변환 (강의 시 transforms.RandomHorizontalFlip(p=1.0) 역할)
img_h = img.transpose(Image.FLIP_LEFT_RIGHT)
img_v = img.transpose(Image.FLIP_TOP_BOTTOM)
img_rot_90 = img.rotate(90, expand=False, fillcolor=(237, 201, 175))

# 2-2. 복합 변환 파이프라인 (강의 시 transforms.Compose 역할)
def custom_transform_pipeline(image):
    transformed = image.copy()
    if random.random() > 0.5:
        transformed = transformed.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() > 0.5:
        transformed = transformed.transpose(Image.FLIP_TOP_BOTTOM)
    
    # 0도에서 360도 사이 랜덤 회전
    angle = random.uniform(0, 360)
    transformed = transformed.rotate(angle, expand=False, fillcolor=(237, 201, 175))
    return transformed

random.seed(42)  # 시연할 때마다 동일한 랜덤 결과를 보여주기 위함
img_pipe1 = custom_transform_pipeline(img)
img_pipe2 = custom_transform_pipeline(img)

# ==========================================
# 3. 터미널 출력 (탐구 시나리오)
# ==========================================
print("\n" + "="*65)
print("🚁 [탐구 미션] 비행기가 거꾸로 날면, 선인장도 거꾸로 자랄까?")
print("="*65)
print("Q1. 여러분이 다운로드한 17,500장의 학습 데이터는 대부분 '1번 Original' 처럼 생겼습니다.")
print("Q2. 그런데 실전(Test)에서 비행기가 서쪽에서 동쪽으로 날며 사진을 찍었다면?")
print("    -> 선인장은 '4번 90도 회전' 한 것처럼 보일 것입니다.\n")
print("Q3. 평생 '1번' 각도의 선인장만 보고 자란 온실 속 화초 같은 AI 모델이,")
print("    과연 '4번' 사진을 보고 선인장이라고 알아맞힐 수 있을까요?")
print("    (아니요! 다른 물체라고 생각할 겁니다!")
print("-" * 65)
print("💡 [Aha-moment: 데이터 증강 (Data Augmentation)]")
print("비싼 돈을 주고 비행기를 다시 띄워 4번 각도의 사진을 찍어올 필요가 없습니다.")
print("우리는 파이토치의 'torchvision.transforms' 파이프라인을 통과시켜,")
print("원본 사진 단 한 장을 5번, 6번처럼 수만 가지의 무작위 각도와 대칭으로 비틀어버릴 것입니다.\n")
print("이것이 바로 항공 사진 분석의 핵심인 '회전 불변성(Rotational Invariance)'의 획득입니다!")
print("="*65 + "\n")

# ==========================================
# 4. 시각화 (Discover)
# ==========================================
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()

titles = ["1. Original\n(How AI sees it first)", 
          "2. Horizontal Flip\n(Plane flying opposite)", 
          "3. Vertical Flip\n(Plane flying upside down?)", 
          "4. 90-Degree Rotation\n(Crosswind path)", 
          "5. Pipeline Random 1\n(Ready for real world)", 
          "6. Pipeline Random 2\n(Infinite variations)"]
images = [img, img_h, img_v, img_rot_90, img_pipe1, img_pipe2]

for ax, t, i in zip(axes, titles, images):
    ax.imshow(i)
    ax.set_title(t, fontweight='bold', pad=15, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.show()
