import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def main():
    # --- Image Classifier Model ---
    class ImageClassifier(nn.Module):
        def __init__(self, num_classes=2):
            super(ImageClassifier, self).__init__()
            self.model = models.resnet18(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    # --- Paths ---
    data_dir = r'D:\KANGROO\TRAINING_AND_VALIDATION'
    save_path = r"D:\KANGROO\SAFARI_GLEN_24005636.pth"

    # --- Data transforms ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
    }

    # --- Load datasets ---
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print(f"🧠 Classes Detected: {class_names}")
    print(f"📦 Train Size: {dataset_sizes['train']} | Val Size: {dataset_sizes['val']}")

    # --- Initialize model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageClassifier(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # --- TensorBoard ---
    writer = SummaryWriter(log_dir='./runs/humans_training')

    # --- Training Loop ---
    num_epochs = 10
    best_val_acc = 0.0
    patience = 5
    counter = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("⏹️ Early stopping triggered.")
            break

        print(f"\n📚 Epoch {epoch + 1}/{num_epochs}")
        print('-' * 30)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            pbar = tqdm(dataloaders[phase], desc=f"[{phase.upper()}] Epoch {epoch + 1}")
            sample_counter = 0

            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                sample_counter += inputs.size(0)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                pbar.set_postfix(loss=loss.item())

            # Ensure full data passed through
            assert sample_counter == dataset_sizes[phase], f"⚠️ Not all {phase} data seen! Expected: {dataset_sizes[phase]}, Got: {sample_counter}"

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
            epoch_rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)

            print(f"{phase.capitalize()} ➤ Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Precision: {epoch_prec:.4f} | Recall: {epoch_rec:.4f}")

            writer.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)
            writer.add_scalar(f'{phase}/Precision', epoch_prec, epoch)
            writer.add_scalar(f'{phase}/Recall', epoch_rec, epoch)

            # --- Early Stopping Check ---
            if phase == 'val':
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    torch.save(model.state_dict(), save_path)
                    print("✅ Best model saved!")
                    counter = 0
                else:
                    counter += 1
                    print(f"⚠️ No improvement. Early stop counter: {counter}/{patience}")
                    if counter >= patience:
                        early_stop = True

    writer.close()
    print("🏁 Training complete!")

if __name__ == '__main__':
    main()
