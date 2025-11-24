import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import random
import time
import matplotlib.pyplot as plt
import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = './models/best_tiny_imagenet_model.pth'
TRAIN_DIR = './dataset/tiny-imagenet-200/train'
WORDS_FILE = './dataset/tiny-imagenet-200/words.txt'
TEST_DIR = './dataset/tiny-imagenet-200/test/images'
WANDB_PROJECT = os.environ.get('WANDB_EVAL_PROJECT', 'tiny-imagenet-eval')
WANDB_RUN_NAME = os.environ.get('WANDB_EVAL_RUN_NAME')

print(f"Using device: {device}")


def create_id_to_name_map(train_path, words_file):
    print("Building label map...")
    
    if not os.path.exists(train_path):
        print(f"Warning: '{train_path}' not found. Cannot map IDs to names.")
        return None

    # Get list of folder names (n01440764, n01629819, etc.) sorted alphabetically
    class_folders = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    
    idx_to_wnid = {i: wnid for i, wnid in enumerate(class_folders)}
    

    wnid_to_name = {}
    if os.path.exists(words_file):
        with open(words_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    wnid_to_name[parts[0]] = parts[1]
    else:
        print(f"Warning: '{words_file}' not found. Labels will be ID codes only.")

    idx_to_name = {}
    for idx, wnid in idx_to_wnid.items():
        full_name = wnid_to_name.get(wnid, wnid)
        short_name = full_name.split(',')[0]
        idx_to_name[idx] = short_name
        
    print(f"✓ Map created for {len(idx_to_name)} classes.")
    return idx_to_name

# Generate the map
label_map = create_id_to_name_map(TRAIN_DIR, WORDS_FILE)
num_classes = len(label_map) if label_map else 200

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

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Weights loaded from {MODEL_PATH}")
else:
    print(f"ERROR: {MODEL_PATH} not found. Please train the model first.")
    exit()

model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def predict_and_show(folder_path, num_samples=5):

    local_rng = random.Random(int(time.time() * 1000000) % (2**32))
    
    # Find all image files
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not all_files:
        print(f"No images found in {folder_path}")
        return

    # Randomly select images (different selection each run)
    num_to_select = min(num_samples, len(all_files))
    selected_files = local_rng.sample(all_files, num_to_select)
    # Shuffle the order for random display arrangement
    local_rng.shuffle(selected_files)
    print(f"Randomly selected {num_to_select} images from {len(all_files)} total images")
    print(f"Selected files: {[f[:20] + '...' if len(f) > 20 else f for f in selected_files]}")

    wandb_run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "num_samples": num_samples,
            "device": device,
            "model_path": MODEL_PATH,
        },
    )

    # Setup Plot
    plt.figure(figsize=(15, 6))
    
    for i, file_name in enumerate(selected_files):
        # Load
        img_path = os.path.join(folder_path, file_name)
        image = Image.open(img_path).convert('RGB')
        
        # Predict
        input_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.nn.Softmax(dim=1)(logits)
            pred_prob, pred_id = torch.max(probs, 1)
            
            idx = pred_id.item()
            conf = pred_prob.item()

        # Get Name
        if label_map:
            class_name = label_map.get(idx, "Unknown")
        else:
            class_name = f"Class {idx}"

        wandb.log({
            "sample_image": wandb.Image(image, caption=f"{class_name} ({conf*100:.1f}%)"),
            "predicted_class": class_name,
            "predicted_id": idx,
            "confidence": conf,
            "file_name": file_name,
        })

        # Visualize
        ax = plt.subplot(1, num_samples, i + 1)
        plt.imshow(image)
        
        # Title with Name and Confidence
        title_text = f"{class_name}\n({conf*100:.1f}%)"
        plt.title(title_text, color='blue', fontsize=12, fontweight='bold')
        plt.axis("off")
        
        print(f"File: {file_name} -> {class_name} (ID: {idx})")

    plt.tight_layout()
    plt.show()
    wandb.finish()

# Run it
predict_and_show(TEST_DIR, num_samples=5)