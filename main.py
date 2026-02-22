import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader


DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 102
BATCH_SIZE    = 32
STAGE1_EPOCHS = 3
STAGE2_EPOCHS = 4
LR_HEAD       = 0.001
LR_FINETUNE   = 0.0001
DATA_DIR      = "./flowers_data"


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])


def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES)
    )
    return model


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")


def run_epoch(model, loader, criterion, optimizer=None, is_train=True):
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train and optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), 100.0 * correct / total


def train_stage(model, stage_name, epochs, loader_tr, loader_val,
                criterion, optimizer, scheduler=None):
    print(f"\n--- {stage_name} ---")
    count_trainable(model)

    best_val_acc = 0.0
    best_path    = f"best_{stage_name.split()[1].lower()}_model.pth"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc   = run_epoch(model, loader_tr, criterion,
                                       optimizer, is_train=True)
        val_loss, val_acc = run_epoch(model, loader_val, criterion,
                                       is_train=False)
        if scheduler:
            scheduler.step()

        mark = "*" if val_acc > best_val_acc else " "
        print(f"  [{mark}] Epoch {epoch}/{epochs}  "
              f"Train: {tr_acc:.1f}%  Val: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)

    print(f"  Best val acc: {best_val_acc:.2f}%")
    return best_path


if __name__ == '__main__':

    print(f"Device: {DEVICE}  |  Classes: {NUM_CLASSES}\n")

    train_data = datasets.Flowers102(root=DATA_DIR, split='train',
                                      download=True, transform=train_transform)
    val_data   = datasets.Flowers102(root=DATA_DIR, split='val',
                                      download=True, transform=val_transform)
    test_data  = datasets.Flowers102(root=DATA_DIR, split='test',
                                      download=True, transform=val_transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"Train: {len(train_data)}  Val: {len(val_data)}  Test: {len(test_data)}\n")

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Stage 1: only the new classification head is trained
    optimizer_s1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD
    )
    scheduler_s1 = optim.lr_scheduler.StepLR(optimizer_s1, step_size=2, gamma=0.5)

    best_s1 = train_stage(
        model, "Stage 1 – Head Only",
        STAGE1_EPOCHS, train_loader, val_loader,
        criterion, optimizer_s1, scheduler_s1
    )

    # Stage 2: unfreeze the last residual block for deeper fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    optimizer_s2 = optim.Adam([
        {"params": model.layer4.parameters(), "lr": LR_FINETUNE},
        {"params": model.fc.parameters(),     "lr": LR_HEAD * 0.5}
    ])
    scheduler_s2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=STAGE2_EPOCHS
    )

    best_s2 = train_stage(
        model, "Stage 2 – Fine-Tuning",
        STAGE2_EPOCHS, train_loader, val_loader,
        criterion, optimizer_s2, scheduler_s2
    )

    # Final evaluation on the test set
    print("\n--- Test Evaluation ---")
    model.load_state_dict(torch.load(best_s2, map_location=DEVICE))
    model.eval()

    top1_c, top5_c, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)

            _, top1_preds = outputs.max(1)
            top1_c += top1_preds.eq(labels).sum().item()

            _, top5_preds = outputs.topk(5, dim=1)
            for i, label in enumerate(labels):
                if label in top5_preds[i]:
                    top5_c += 1

            total += labels.size(0)

    print(f"  Top-1 Accuracy: {100.*top1_c/total:.2f}%")
    print(f"  Top-5 Accuracy: {100.*top5_c/total:.2f}%")
    print(f"  Model saved as: {best_s2}")
