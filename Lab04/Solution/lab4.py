import os
import re
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from datetime import datetime

# 1. dataset cu perechi de imagini
class TimeLapseDataset(Dataset):
    def __init__(self, folder_path, transform=None):

        self.folder_path = folder_path
        self.transform = transform
        self.image_pairs = [] # stocare perechi de imagini si time skip
        self.image_cache = {}

        self._parse_dataset() # citire si organizare dataset

    def _parse_dataset(self):
        # parcurgere foldere
        location_folders = os.listdir(self.folder_path)
        # pattern pt extragere a timpului si numelui fisierului
        time_pattern = re.compile(r'global_monthly_(\d{4}_\d{2})')

        for location in location_folders:
            location_path = os.path.join(self.folder_path, location, 'images')
            if not os.path.isdir(location_path):
                continue

            images = sorted(os.listdir(location_path))
            time_image_pairs = []
            # extragere timp si creare perechi
            for image_name in images:
                match = time_pattern.search(image_name)

                if match:
                    # extragerea timpului
                    time_str = match.group(1)
                    time_value = datetime.strptime(time_str, "%Y_%m")
                    time_image_pairs.append((time_value, image_name))
            # sorate imgs in ordine cronologica
            time_image_pairs.sort()
            # formarea perechilor de imagini
            for i in range(len(time_image_pairs) - 1):
                start_time, start_image = time_image_pairs[i]

                for j in range(i + 1, len(time_image_pairs)):
                    end_time, end_image = time_image_pairs[j]
                    # calculare diferenta intre luni
                    time_skip = (end_time.year - start_time.year) * 12 + (end_time.month - start_time.month)
                    # salvare pereche de inagini si time skip
                    if time_skip > 0:
                        self.image_pairs.append((os.path.join(location_path, start_image),
                                                 os.path.join(location_path, end_image),
                                                 time_skip))

        self.dataset_size = len(self.image_pairs) # dimens dataset

    # caching imagini
    def _load_image(self, image_path):
        if image_path in self.image_cache:
            return self.image_cache[image_path]

        image = Image.open(image_path).convert("RGB")

        # adaugare imagine in cache
        self.image_cache[image_path] = image

        return image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        start_image_path, end_image_path, time_skip = self.image_pairs[idx]

        # incarcare imagini
        start_image = self._load_image(start_image_path)
        end_image = self._load_image(end_image_path)

        if self.transform:
            # aplicare rotatie cu verificare de a fi aceeasi rotatie (random augmentation no.2)
            angle = random.uniform(-30, 30)
            transform = transforms.Compose([
                transforms.RandomRotation((angle, angle)),
                *self.transform.transforms
            ])
            start_image = transform(start_image)
            end_image = transform(end_image)
        else:
            start_image = transforms.ToTensor()(start_image)
            end_image = transforms.ToTensor()(end_image)

        # returnare tuplu cerut
        return start_image, end_image, time_skip


dataset_path = "../Dataset"

# definire transformari
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(), # (random augmentation no.1, no.2 in getitem)
    transforms.ColorJitter(brightness=0.5, contrast=0.5),
    transforms.ToTensor()
])

dataset = TimeLapseDataset(folder_path=dataset_path, transform=transform)
# afisare lungime dataset
print(f"Dataset size: {len(dataset)}")

# afisarea unui esantion
sample = dataset[1]
print(f"Start image shape: {sample[0].shape}, End image shape: {sample[1].shape}, Time skip: {sample[2]}")


# 2. definirea proportiilor de impartire a datasetului
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

# impartirea efectiva a datasetului: antrenare, validare, testare
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# definire batch
batch_size = 32

# creare dataloadere pt antr, val si testare
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# afisare dimensiuni loadere
print(f"Train set size: {len(train_loader.dataset)}")
print(f"Validation set size: {len(val_loader.dataset)}")
print(f"Test set size: {len(test_loader.dataset)}")

train_sample = next(iter(train_loader))
print(f"Start image batch shape: {train_sample[0].shape}")
print(f"End image batch shape: {train_sample[1].shape}")
print(f"Time skips: {train_sample[2]}")


# 3. clasa model autoencoder pt reconstructia imaginii
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()

        # encoder: reducere dimensiune si capturare caracterictici importatnte
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # (batch, 3, 128, 128) -> (batch, 64, 64, 64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (batch, 128, 32, 32)
            nn.ReLU(),
        )

        # decoder: reconctructie imagine la dimens originala
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (batch, 64, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (batch, 3, 128, 128)
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# functia de antrenare pentru o epoca
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    for start_images, end_images, _ in train_loader:
        start_images, end_images = start_images.to(device), end_images.to(device)

        # forward pass
        outputs = model(start_images)
        loss = criterion(outputs, end_images)

        # backward pass + optimizare
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # calcul acuratețe
        correct_pixels += torch.sum(torch.abs(outputs - end_images) < 0.1).item()  # pixeli cu eroare < 0.1
        total_pixels += end_images.numel()

    accuracy = correct_pixels / total_pixels
    return running_loss / len(train_loader), accuracy


# functie de validare pt o epoca
def val(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for start_images, end_images, _ in val_loader:
            start_images, end_images = start_images.to(device), end_images.to(device)

            outputs = model(start_images)
            loss = criterion(outputs, end_images)

            running_loss += loss.item()

            # calcul acuratețe
            correct_pixels += torch.sum(torch.abs(outputs - end_images) < 0.1).item()
            total_pixels += end_images.numel()

    accuracy = correct_pixels / total_pixels
    return running_loss / len(val_loader), accuracy

# functie de testare pentru o epoca
def test(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_pixels = 0
    correct_pixels = 0

    with torch.no_grad():
        for start_images, end_images, _ in test_loader:
            start_images, end_images = start_images.to(device), end_images.to(device)

            outputs = model(start_images)
            loss = criterion(outputs, end_images)

            running_loss += loss.item()

            # calcul acuratețe
            correct_pixels += torch.sum(torch.abs(outputs - end_images) < 0.1).item()
            total_pixels += end_images.numel()

    accuracy = correct_pixels / total_pixels
    return running_loss / len(test_loader), accuracy


def run(model, train_loader, val_loader, test_loader, criterion, optimizer, device, n_epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(n_epochs):
        # antrenare
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        # validare
        val_loss, val_acc = val(model, val_loader, criterion, device)

        # salvare loss + acurateti
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # afisare metrici antrenare si validare
        print(
            f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # grafice loss-uri si acurateti
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    # testare finala + afisare metrici testare
    test_loss, test_acc = test(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


criterion = nn.MSELoss()  # mse pt reconstructie
n_epochs = 10
# init cpu si gpu
device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# daca gpu e disp rulam pe cuda
if torch.cuda.is_available():
    print("\nRunning on GPU:")
    model_gpu = SimpleAutoencoder().to(device_gpu)
    optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=1e-3)

    # cuda
    run(model_gpu, train_loader, val_loader, test_loader, criterion, optimizer_gpu, device_gpu, n_epochs)
else:
    print("\nGPU not available, only CPU used.")

# rulare cpu
print("Running on CPU:")
model_cpu = SimpleAutoencoder().to(device_cpu)
optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=1e-3)

# rulare fortata pe gpu
run(model_cpu, train_loader, val_loader, test_loader, criterion, optimizer_cpu, device_cpu, n_epochs)