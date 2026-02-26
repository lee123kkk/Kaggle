import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision

train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)

# 1. 시각화 함수 정의
def plot_learning_curves(history):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # [왼쪽] 손실(Loss) 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # [오른쪽] 정확도(Accuracy) 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Validation Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 2. (사례 상황) 과적합이 발생한 가상 데이터 생성
# 상황: 훈련셋 성능은 계속 좋아지는데(Loss 감소), 검증셋 성능은 나빠짐(Loss 증가)
epochs_num = 20
history_mock = {
    'train_loss': np.linspace(0.8, 0.1, epochs_num),            # 0.8 -> 0.1 (쭉 좋아짐)
    'val_loss': np.concatenate([
        np.linspace(0.85, 0.4, 8),   # 8에폭까지는 같이 좋아지다가
        np.linspace(0.4, 1.2, 12)    # 9에폭부터 갑자기 나빠짐 (Overfitting!)
    ]),
    'train_acc': np.linspace(70, 99, epochs_num),               # 쭉 좋아짐
    'val_acc': np.concatenate([
        np.linspace(68, 85, 8),      # 같이 좋아지다가
        np.linspace(85, 82, 12)      # 정체되거나 떨어짐
    ])
}

# 3. 그래프 출력
plot_learning_curves(history_mock)
#===================================================================
# 학습 곡선 구현 
# 과적합 상황을 가정한 가상의 데이터를 넣으 그래프를 그린다.
# epoch에서 초반에는 파란선과 빨간 선이 같이 내려간다, 모델이 정상적으로 학습하고 있다.
# Loss에서 중반 이후로는 파란선을 내려가는데 빨간서은 올라간다.
# acuracy에서 파란선은 중간에 멈추는 과적합 상태이다.
# 조기 종료를 하고 데이터 증강을 하는 등의 수정이 필요하다.

