
# In[1]:
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jay7080dev/rice-plant-diseases-dataset")

print("Path to dataset files:", path)

# In[2]:
from pathlib import Path

root = Path(path)   # path = "/root/.cache/kagglehub/datasets/noulam/tomato/versions/1"

print("Dataset root:", root)
!ls -R {root}


# In[3]:
import pandas as np
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[4]:
print(torch.__version__)
print(torchvision.__version__)

# In[5]:
from pathlib import Path

root = Path(path)
dataset_root = root / "rice leaf diseases dataset"
print("Dataset root:", dataset_root)



# In[6]:
all_images = []
extensions = ["*.jpg", "*.JPG", "*.png", "*.PNG"]

for cls in dataset_root.iterdir():
    if cls.is_dir():
        for ext in extensions:
            all_images.extend(list(cls.glob(ext)))

print("Total images found:", len(all_images))
print(all_images[:5])

# In[7]:
import random
from PIL import Image
import matplotlib.pyplot as plt


assert len(all_images) > 0


random_image_path = random.choice(all_images)


image_class_name = random_image_path.parent.stem


opened_image = Image.open(random_image_path).convert("RGB")


plt.figure(figsize=(6, 6))
plt.imshow(opened_image)
plt.title(f"Class: {image_class_name}")
plt.axis("off")
plt.show()

print(f"Random image path: {random_image_path}")
print(f"Image class: {image_class_name}")


# In[8]:
#turn the image into array
img_as_array = np.array(opened_image)
#plot the image
plt.figure(figsize=(12,6))
plt.imshow(img_as_array)
plt.xlabel(image_class_name)
plt.title("image after transform")
plt.axis(False)

# In[9]:
image_data_train_transform = transforms.Compose([
    #resize our image to 224,224
    transforms.RandomResizedCrop(size=(224,224),scale=(0.8,1.0)),#zoom in ramdomly
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),#add vertical flip with a low probability
    transforms.RandomRotation(degrees=30),
    #turn the data into a torch.tensor
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2)) # random erase parts of the image to make it harder
])
image_data_test_transform = transforms.Compose([
    transforms.Resize(size=(224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# In[10]:
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# === 1. Correct dataset path ===
dataset_root = Path(path) / "rice leaf diseases dataset"
print("Dataset root:", dataset_root)

# === 2. Use your transformations ===
train_data = datasets.ImageFolder(
    root=dataset_root,
    transform=image_data_train_transform
)

test_data = datasets.ImageFolder(
    root=dataset_root,
    transform=image_data_test_transform
)

# === 3. DataLoader settings ===
batch_size = 64
num_workers = 0

train_dataloader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

# === 4. Get a batch ===
images, labels = next(iter(train_dataloader))
img_tensor = images[0]

# === 5. Un-normalize parameters (ImageNet mean/std) ===
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# === 6. Unnormalize function ===
def unnormalize(img_tensor, mean, std):
    img_tensor = img_tensor.clone()
    for t, m, s in zip(img_tensor, mean, std):
        t.mul_(s).add_(m)
    return img_tensor

# === 7. Prepare image for plotting ===
img_unnormalized = unnormalize(img_tensor, mean, std)
img_to_plot = img_unnormalized.permute(1, 2, 0).numpy().clip(0, 1)

# === 8. Plot ===
plt.figure(figsize=(8,8))
plt.imshow(img_to_plot)
plt.title(f"Class: {train_data.classes[labels[0]]}")
plt.axis("off")
plt.show()


# In[11]:
from pathlib import Path

root = Path("/root/.cache/kagglehub/datasets/jay7080dev/rice-plant-diseases-dataset/versions/1")
dataset_root = root / "rice leaf diseases dataset"

all_images = []
extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]

for class_folder in dataset_root.iterdir():
    if class_folder.is_dir():
        for ext in extensions:
            all_images.extend(list(class_folder.glob(ext)))

print("Total images found:", len(all_images))
print("Example:", all_images[:5])



# In[12]:
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch

def plot_transformed_image(image_paths, transform, n=3, seed=None):
    if seed:
        random.seed(seed)

    sample_paths = random.sample(image_paths, k=n)

    for image_path in sample_paths:

        img = Image.open(image_path).convert("RGB")

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))


        ax[0].imshow(img)
        ax[0].set_title("Original")
        ax[0].axis("off")


        transformed = transform(img)

        if isinstance(transformed, torch.Tensor):
            transformed = transformed.permute(1, 2, 0).cpu().numpy()

        transformed = np.clip(transformed, 0, 1)

        ax[1].imshow(transformed)
        ax[1].set_title("Transformed")
        ax[1].axis("off")

        plt.suptitle(f"Class: {image_path.parent.stem}", fontsize=14)
        plt.show()


# In[13]:
print(len(all_images))


# In[14]:
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

# Rice dataset root
root = Path("/root/.cache/kagglehub/datasets/jay7080dev/rice-plant-diseases-dataset/versions/1")
dataset_root = root / "rice leaf diseases dataset"

print("Dataset root:", dataset_root)

# === Create Dataset ===
full_dataset = datasets.ImageFolder(
    root=dataset_root,
    transform=image_data_train_transform
)

print("Total Images:", len(full_dataset))
print("Classes:", full_dataset.classes)



# In[15]:
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

# ==== 1. Load full dataset ====
full_dataset = datasets.ImageFolder(
    root=dataset_root,
    transform=image_data_train_transform
)

# ==== 2. Split indices ====
indices = list(range(len(full_dataset)))
train_idx, valid_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)

train_dataset = Subset(full_dataset, train_idx)
valid_dataset = Subset(full_dataset, valid_idx)

# valid dataset 应该用 test_transform
valid_dataset.dataset.transform = image_data_test_transform

# ==== 3. Dataloaders ====
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader  = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=0)

print("Train Images:", len(train_dataset))
print("Valid/Test Images:", len(valid_dataset))
print("Classes:", full_dataset.classes)

# In[16]:
len(train_dataset), len(valid_dataset)


# In[17]:
img, label = train_dataset[0][0], valid_dataset[0][1]
print(f"image tensor:\n {img}")
print(f"image shape: {img.shape}")
print(f"image datatype: {img.dtype}")
print(f"image label: {label}")
print(f"label datatype: {type(label)}")

# In[18]:
import torch
import matplotlib.pyplot as plt

def imshow_tensor(img_tensor):
    """
    Un-normalize & display a tensor image.
    Works with RICE DATASET since we use ImageNet normalization.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    img = img_tensor.clone()
    img = img * std + mean     # unnormalize
    img = img.permute(1,2,0)   # CHW → HWC
    img = torch.clamp(img, 0, 1)

    plt.imshow(img.numpy())
    plt.axis("off")


# In[19]:
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from pathlib import Path
import numpy as np
import torch

# === 1. Load RICE DATASET ===
root = Path("/root/.cache/kagglehub/datasets/jay7080dev/rice-plant-diseases-dataset/versions/1")
dataset_root = root / "rice leaf diseases dataset"

full_dataset = datasets.ImageFolder(
    root=dataset_root,
    transform=image_data_train_transform   # train transform for now
)

total_size = len(full_dataset)
print(f"Total images in full_dataset: {total_size}")

# === 2. Split 70 / 15 / 15 ===
train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

print(f"Training size:   {train_size}")
print(f"Validation size: {val_size}")
print(f"Test size:       {test_size}")

# === 3. Shuffle indices ===
indices = list(range(total_size))
np.random.seed(42)
np.random.shuffle(indices)

train_idx = indices[:train_size]
val_idx   = indices[train_size : train_size + val_size]
test_idx  = indices[train_size + val_size :]

# === 4. Create Subsets (Rice version) ===
train_dataset = Subset(full_dataset, train_idx)
val_dataset   = Subset(full_dataset, val_idx)
test_dataset  = Subset(full_dataset, test_idx)

# Validation/Test 应使用 test_transform
val_dataset.dataset.transform  = image_data_test_transform
test_dataset.dataset.transform = image_data_test_transform

print("Split finished without leakage.")

# === 5. Dataloaders ===
BATCH_SIZE = 16

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0, pin_memory=True)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0, pin_memory=True)

print("Train batches:", len(train_dataloader))
print("Val batches:", len(val_dataloader))
print("Test batches:", len(test_dataloader))

print("Classes:", full_dataset.classes)


# In[20]:
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

def improve_resnet_model(num_classes: int):
    """
    Creates a ResNet-50 model with a frozen base and a custom classifier head.

    Args:
        num_classes (int): The number of output classes for the model.
    """
    #load a pre-trained resnet50 model
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    #freeze the base model parameters
    for param in model.parameters():
        param.requires_grad = False

    #Get the number of input features from the original fully connected layer
    num_ftrs = model.fc.in_features

    #Replace the final layer with a new, more powerful classifier head (Improved)
    model.fc = nn.Sequential( # Use nn.Sequential for multiple layers
        nn.Linear(num_ftrs, 512),   # Add an intermediate layer
        nn.ReLU(),                 # Add a non-linear activation
        nn.Dropout(p=0.5),         # Add dropout for regularization
        nn.Linear(512, num_classes) # The final output layer
    )
    return model

# In[21]:
#create instance of tinyvgg
torch.manual_seed(42)
# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"
# insatantial the model
model = improve_resnet_model(num_classes=len(train_data.classes)).to(device)
print(model)


# In[22]:
#Get a single image batch
image_batch, label_batch = next(iter(train_dataloader))
image_batch.shape, label_batch.shape
#batch size will now be 1, you can change the batch size if you like
print(f"image shape: {image_batch.shape} -> [batch_size, color_channels, height, width]")
print(f"label shape: {label_batch.shape}")

# In[23]:
#try a forward pass
model(image_batch.to(device))

# In[24]:
try:
  import torchinfo
except:
  !pip install torchinfo
  import torchinfo
from torchinfo import summary
summary(model, input_size=[32, 3, 224, 224])

# In[25]:
def train_step(model: torch.nn.Module,
               train_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):

    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # Forward
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y)

    # Average per batch
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)

    return train_loss, train_acc


# In[26]:
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):

    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_labels = y_pred.argmax(dim=1)
            test_acc += (y_pred_labels == y).sum().item() / len(y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

    return test_loss, test_acc


# In[27]:
from tqdm.auto import tqdm
import torch

def train_final(model: torch.nn.Module,
                train_dataloader: torch.utils.data.DataLoader,
                test_dataloader: torch.utils.data.DataLoader,   # <-- val_dataloader for rice
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                loss_fn: torch.nn.Module,
                epochs: int,
                device: torch.device,
                patience: int = 5):

    print(">>> Starting training with early stopping and scheduler...")

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = "best_model.pth"

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    for epoch in tqdm(range(epochs)):

        # ---- Training ----
        train_loss, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )

        # ---- Validation ----
        val_loss, val_acc = test_step(
            model, test_dataloader, loss_fn, device
        )

        # ---- Scheduler ----
        scheduler.step(val_loss)

        # ---- Logging ----
        print(f"\nEpoch {epoch+1}/{epochs}"
              f" | Train Acc: {train_acc:.4f}"
              f" | Val Acc: {val_acc:.4f}"
              f" | Val Loss: {val_loss:.4f}")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        # ---- Early Stopping ----
        if val_loss < best_val_loss:
            print(f"✅ Improved Val Loss: {best_val_loss:.4f} → {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), best_model_path)
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("⏹ Early stopping triggered!")
            break

    # ---- Load best model ----
    model.load_state_dict(torch.load(best_model_path))
    print("🏆 Training complete. Best model loaded.")

    return model, history


# In[28]:
import torch.nn as nn
import torch.optim as optim

# Loss function
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=1e-4)


scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)


# In[29]:
model, history = train_final(
    model,
    train_dataloader=train_dataloader,   # rice 70%
    test_dataloader=val_dataloader,      # rice 15%
    optimizer=optimizer,
    scheduler=scheduler,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
    patience=5
)


# In[30]:
import matplotlib.pyplot as plt

plt.plot(history["train_loss"])
plt.plot(history["val_loss"])
plt.legend(["train", "val"])
plt.title("Loss Curve")
plt.show()

plt.plot(history["train_acc"])
plt.plot(history["val_acc"])
plt.legend(["train", "val"])
plt.title("Accuracy Curve")
plt.show()


# In[31]:
import torch

def get_all_preds(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)


# In[32]:
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

labels, preds = get_all_preds(model, test_dataloader, device)

cm = confusion_matrix(labels, preds)
class_names = full_dataset.classes   # ['Bacterialblight', 'Brownspot', 'Leafsmut']

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix (Rice Disease Classification)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# In[33]:
accuracy = (preds == labels).mean()
print(f"Test Accuracy: {accuracy*100:.2f}%")


# In[34]:
for i, cls in enumerate(class_names):
    cls_idx = np.where(labels == i)[0]
    cls_acc = (preds[cls_idx] == i).mean()
    print(f"{cls}: {cls_acc*100:.2f}%")


# In[35]:
from sklearn.metrics import classification_report

print(classification_report(labels, preds, target_names=class_names))
