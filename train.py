import glob
import os
import random
import time

import torch
import torch.optim as optim
from datasets import load_dataset
from torch.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ColorJitter, RandomErasing
from tqdm import tqdm

from nbb import UpsampleConcatClassifier  # 사용자 정의 모델 import

# -----------------------------------
# 설정: 시드 고정
# -----------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# -----------------------------------
# 사전 학습된 모델 경로 (없으면 None)
# -----------------------------------
PRETRAINED_MODEL_PATH = None  # 또는 경로 문자열
# PRETRAINED_MODEL_PATH = "./checkpoints/save_last.pth"


# -------------------------
# Mixup 함수
# -------------------------
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ----------------------------
# 전처리 정의
# ----------------------------

class RandomizedGaussianBlur:
    def __init__(self, kernel_sizes=(3, 5), sigma=(0.1, 1.5), p=0.3):
        self.kernel_sizes = kernel_sizes
        self.sigma = sigma
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            k = random.choice(self.kernel_sizes)
            return transforms.GaussianBlur(kernel_size=k, sigma=self.sigma)(img)
        return img


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(320, scale=(0.5, 1.0), ratio=(0.75, 1.33)),  # 랜덤 크롭 + 다양한 비율
    transforms.RandomHorizontalFlip(p=0.5),  # 좌우 반전
    transforms.RandomApply([ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),  # 색상 변화
    RandomizedGaussianBlur(p=0.3),  # 블러
    transforms.RandomApply([transforms.RandomPerspective(distortion_scale=0.1)], p=0.3),  # 원근 왜곡
    transforms.ToTensor(),
    RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((350, 350)),
    transforms.CenterCrop((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ----------------------------
# transform 함수 정의
# ----------------------------
def transform_train(example):
    try:
        image = example["image"].convert("RGB")
    except Exception:
        return None
    example["pixel_values"] = train_transform(image)
    return example


def transform_val(example):
    try:
        image = example["image"].convert("RGB")
    except Exception:
        return None
    example["pixel_values"] = val_transform(image)
    return example


if __name__ == "__main__":
    # ----------------------------
    # TensorBoard 설정
    # ----------------------------
    writer = SummaryWriter(log_dir="runs/exp1")

    # ----------------------------
    # 데이터셋 로드 및 전처리
    # ----------------------------
    train_ds = load_dataset("food101", split="train")
    val_ds = load_dataset("food101", split="validation")

    # 일부 샘플만 선택 (shuffle 먼저 하면 더 랜덤)
    # train_ds = train_ds.shuffle(seed=SEED).select(range(400))
    # val_ds = val_ds.shuffle(seed=SEED).select(range(100))

    train_ds = train_ds.map(transform_train, num_proc=1)
    train_ds = train_ds.filter(lambda x: x is not None)
    val_ds = val_ds.map(transform_val, num_proc=1)
    val_ds = val_ds.filter(lambda x: x is not None)

    train_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    val_ds.set_format(type='torch', columns=['pixel_values', 'label'])

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2)

    # ----------------------------
    # 모델/옵티마이저/스케줄러/스칼러 설정
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UpsampleConcatClassifier(num_classes=101).to(device)

    if PRETRAINED_MODEL_PATH is not None and os.path.exists(PRETRAINED_MODEL_PATH):
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=device))
        print(f"Loaded pretrained model from {PRETRAINED_MODEL_PATH}")

    criterion_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=30)
    scaler = GradScaler(device='cuda')


    # ----------------------------
    # 학습 및 평가 함수 정의
    # ----------------------------
    def train_epoch(model, dataloader, optimizer, device, scaler, use_mixup=True):
        model.train()
        total_loss = total_correct = total_samples = 0
        pbar = tqdm(dataloader, desc="Training", leave=False)

        for batch in pbar:
            inputs = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                if use_mixup:
                    inputs_mixed, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
                    outputs = model(inputs_mixed)
                    loss = mixup_criterion(criterion_fn, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            if use_mixup:
                total_correct += ((preds == targets_a) | (preds == targets_b)).sum().item()
            else:
                total_correct += (preds == labels).sum().item()

            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}", "Avg Acc": f"{total_correct / total_samples:.4f}"})

        return total_loss / total_samples, total_correct / total_samples


    def eval_epoch(model, dataloader, device):
        model.eval()
        total_loss = total_correct = total_samples = 0
        start_time = time.perf_counter()

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validating", leave=False)
            for batch in pbar:
                inputs = batch['pixel_values'].to(device)
                labels = batch['label'].to(device)

                outputs = model(inputs)
                loss = criterion_fn(outputs, labels)
                preds = outputs.argmax(dim=1)

                total_loss += loss.item() * inputs.size(0)
                total_correct += (preds == labels).sum().item()
                total_samples += inputs.size(0)
                pbar.set_postfix(
                    {"Batch Loss": f"{loss.item():.4f}", "Avg Acc": f"{total_correct / total_samples:.4f}"})

        end_time = time.perf_counter()
        inference_time = end_time - start_time
        avg_inf_time = inference_time / total_samples
        return total_loss / total_samples, total_correct / total_samples, inference_time, avg_inf_time


    # ----------------------------
    # 전체 학습 실행 + 모델 저장 (Early Stopping, last.pth)
    # ----------------------------
    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0
    no_improve_epochs = 0
    patience = 5
    max_epochs = 30
    epoch = 0

    while epoch < max_epochs:
        epoch += 1
        print(f"\n===== Epoch {epoch} =====")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, scaler, use_mixup=True)
        val_loss, val_acc, val_time, avg_inf_time = eval_epoch(model, val_loader, device)
        scheduler.step()

        # TensorBoard 기록
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(
            f"Val   Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Inference Time: {val_time:.2f}s | Avg Per Image: {avg_inf_time * 1000:.2f}ms")

        # 매 에폭 last 체크포인트
        torch.save(model.state_dict(), "checkpoints/last.pth")

        # best 모델 저장 및 Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_epochs = 0
            for f in glob.glob("checkpoints/best_*.pth"):
                os.remove(f)
            best_path = f"checkpoints/best_epoch{epoch:02d}_acc{val_acc:.4f}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"💾 Saved best model as {os.path.basename(best_path)}.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"⏱ Early stopping at epoch {epoch} after {patience} epochs without improvement.")
                break

    writer.close()
