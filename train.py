import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

BATCH_SIZE = 64
LEARNING_RATE = 0.01  # Start higher because we use a scheduler
EPOCHS = 6
DATA_DIR = './dataset/tiny-imagenet-200'  # Make sure this matches your folder structure
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'tiny-imagenet-training')
WANDB_RUN_NAME = os.environ.get('WANDB_RUN_NAME')


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),  # Flip images horizontally
    transforms.RandomRotation(degrees=10),     # Rotate slightly
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Vary colors
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
])

# No augmentation for validation/test (we want consistent evaluation)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Data...")

train_dir = os.path.join(DATA_DIR, 'train')
val_dir = os.path.join(DATA_DIR, 'val') 
num_classes = 200  # default fallback if dataset fails to load


try:
    training_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(root=val_dir, transform=test_transform)

    # IMPROVEMENT: Add num_workers for faster data loading (if CPU cores available)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=2, pin_memory=True if device == 'cuda' else False)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=2, pin_memory=True if device == 'cuda' else False)
    num_classes = len(training_data.classes)
    print(f"Data Loaded. Classes: {num_classes}")
    print("Using data augmentation for training to improve generalization.")
except Exception as e:
    print(f"Error loading data: {e}")
    print("Ensure your './dataset/tiny-imagenet-200' folder exists and has 'train'/'val' subfolders.")
    exit()


class CustomCNN(nn.Module):
    def __init__(self, num_classes=200, dropout_rate=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CustomCNN(num_classes=num_classes).to(device)

loss_fn = nn.CrossEntropyLoss()



# Option 1: Adam (usually better for image classification)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # weight_decay = L2 regularization

# Option 2: SGD with momentum (original, uncomment if you prefer)
# optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

# IMPROVEMENT: Better learning rate scheduler
# ReduceLROnPlateau reduces LR when validation loss stops improving (more adaptive)
# Or use CosineAnnealingLR for smoother decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
# Alternative: CosineAnnealingLR (smooth cosine decay)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set to train mode
    total_loss = 0.0
    num_batches = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        
        # IMPROVEMENT: Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return total_loss / num_batches  # Return average loss

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    model.eval() # Set to eval mode
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = correct / size
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy  # Return loss & accuracy to track best model


if __name__ == '__main__':
    best_accuracy = 0.0
    patience_counter = 0
    early_stop_patience = 5  # IMPROVEMENT: Early stopping - stop if no improvement for N epochs
    models_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, 'best_tiny_imagenet_model.pth')

    print("Starting Training...")
    print("=" * 60)
    # Start timing for total training duration
    total_start_time = time.time()
    training_start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training started at: {training_start_time_str}\n")

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "num_classes": num_classes,
            "device": device,
        },
    )
    wandb.watch(model, log="all", log_freq=100)

    for t in range(EPOCHS):
        # Get current learning rate (works for both scheduler types)
        try:
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
        except:
            current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {t+1}/{EPOCHS} (LR: {current_lr:.6f})\n-------------------------------")
        
        start_time = time.time()
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        val_loss, current_accuracy = test_loop(test_dataloader, model, loss_fn)
        
        # Step the scheduler (adapt based on validation accuracy)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(current_accuracy)  # Pass metric to ReduceLROnPlateau
        else:
            scheduler.step()
        
        # IMPROVEMENT: Save only if this is the best model so far
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"--> New Best Model Saved to {best_model_path} (Accuracy: {100*current_accuracy:.1f}%)")
            wandb.save(best_model_path)
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")

        wandb.log({
            "epoch": t + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": current_accuracy,
            "learning_rate": current_lr,
            "patience_counter": patience_counter,
        })
        
        # IMPROVEMENT: Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {t+1} epochs (no improvement for {early_stop_patience} epochs)")
            break
    
        print(f"Epoch time: {time.time() - start_time:.1f}s")
        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}\n")

    print("=" * 60)
    # Calculate total training time
    total_end_time = time.time()
    total_training_time = total_end_time - total_start_time

    # Format time nicely
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    milliseconds = int((total_training_time % 1) * 1000)

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Best Accuracy achieved: {100*best_accuracy:.1f}%")
    print(f"Model saved to: '{best_model_path}'")
    print(f"\nTotal Training Time: {total_training_time:.2f} seconds")
    if hours > 0:
        print(f"  = {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"  = {minutes}m {seconds}s {milliseconds}ms")
    else:
        print(f"  = {seconds}s {milliseconds}ms")
    print(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    wandb.summary['best_accuracy'] = best_accuracy
    wandb.summary['best_model_path'] = best_model_path
    wandb.finish()