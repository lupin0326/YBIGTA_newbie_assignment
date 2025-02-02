import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *
from tqdm import tqdm


NUM_CLASSES = 10  

# CIFAR-10 데이터셋 로드 (Tensor 변환 추가)
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 4픽셀 패딩 후 랜덤 크롭
    transforms.RandomHorizontalFlip(),  # 50% 확률로 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),  # 정규화 추가
])


# CIFAR-10 데이터셋 로드
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet 18 선언하기
## TODO
model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=NUM_CLASSES).to(device)

criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
optimizer: optim.Adam = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Early Stopping 설정
early_stopping_patience = 5  # 5 Epoch 동안 개선되지 않으면 종료
best_loss = float('inf')  # 최상의 Test Loss
patience_counter = 0  # 개선되지 않은 Epoch 수 카운트

# 학습 
def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    model.train()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    progress_bar = tqdm(loader, desc="Training", leave=True)    # 제거

    for batch_idx, (inputs, targets) in enumerate(progress_bar):    # for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total) # 제거

    accuracy: float = 100. * correct / total
    avg_loss = total_loss / len(loader)
    print(f"Train Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    scheduler.step()
    return avg_loss

# 평가 
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> None:
    model.eval()
    total_loss: float = 0
    correct: int = 0
    total: int = 0

    progress_bar = tqdm(loader, desc="Evaluating", leave=True) # 제거

    with torch.no_grad():
        for inputs, targets in progress_bar:  # for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    avg_loss = total_loss / len(loader)
    accuracy: float = 100. * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss

# 학습 및 평가 루프 (Early Stopping 적용)
for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    train_loss = train(model, train_loader, criterion, optimizer, device)
    test_loss = evaluate(model, test_loader, criterion, device)

    # Early Stopping 체크
    if test_loss < best_loss:
        best_loss = test_loss
        patience_counter = 0  # 개선되었으므로 patience 초기화
        print(f"Test Loss Improved: {best_loss:.4f}")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter}/{early_stopping_patience} epochs.")

    # Early Stopping 조건 충족 시 학습 중단
    if patience_counter >= early_stopping_patience:
        print(f"Early Stopping Triggered. Training Stopped.")
        break  # 학습 중단

# 모델 저장
torch.save(model.state_dict(), "resnet18_checkpoint.pth")
print(f"Model saved to resnet18_checkpoint.pth")
