import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from openemotion.config import CLASS_NAMES, IMG_SIZE
from openemotion.model import create_model


def main():
    parser = argparse.ArgumentParser(description="Train OpenEmotion model on FER-2013")
    parser.add_argument("--data", required=True, help="FER-2013 root directory containing train and test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="models/emotion_model_best.pth")
    parser.add_argument("--use-class-weights", action="store_true", help="Use weighted CrossEntropyLoss for imbalanced dataset")
    args = parser.parse_args()

    root = Path(args.data)
    train_dir = root / "train"
    val_dir = root / "test"

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2 if torch.cuda.is_available() else 0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    val_loader = None
    if val_dir.exists():
        val_dataset = datasets.ImageFolder(str(val_dir), transform=val_transform)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2 if torch.cuda.is_available() else 0,
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Training on device: {device}")

    model = create_model(len(CLASS_NAMES)).to(device)

    if args.use_class_weights:
        targets = [label for _, label in train_dataset.samples]
        class_weights = compute_class_weight("balanced", classes=np.unique(targets), y=targets)
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        print("Using class-weighted CrossEntropyLoss")
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / total

        val_acc = 0.0
        val_loss = 0.0
        if val_loader:
            model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * images.size(0)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            val_loss = val_running_loss / val_total
            scheduler.step(val_acc)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if val_acc > best_val_acc or not val_loader:
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model (Val Acc: {val_acc:.4f}) to {output_path}")

    print("Training finished successfully.")


if __name__ == "__main__":
    main()
