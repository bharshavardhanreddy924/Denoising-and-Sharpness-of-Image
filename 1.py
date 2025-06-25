import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CustomDenoiseCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17, base_features=64):
        super(CustomDenoiseCNN, self).__init__()
        
        kernel_size = 3
        padding = 1
        
        layers = [
            nn.Conv2d(in_channels=channels, out_channels=base_features, 
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True)
        ]
        
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(in_channels=base_features, out_channels=base_features, 
                          kernel_size=kernel_size, padding=padding, bias=False),
                nn.BatchNorm2d(base_features),
                nn.ReLU(inplace=True)
            ])
        
        layers.append(
            nn.Conv2d(in_channels=base_features, out_channels=channels, 
                      kernel_size=kernel_size, padding=padding, bias=False)
        )
        
        self.denoiser = nn.Sequential(*layers)
    
    def forward(self, x):
        noise = self.denoiser(x)
        return x - noise

class SIDDDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        self.noisy_images = []
        self.clean_images = []
        
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if 'NOISY' in file:
                    self.noisy_images.append(os.path.join(subdir, file))
                elif 'GT' in file:
                    self.clean_images.append(os.path.join(subdir, file))
        
        self.noisy_images.sort()
        self.clean_images.sort()

    def __len__(self):
        return len(self.noisy_images)

    def __getitem__(self, idx):
        noisy_img = Image.open(self.noisy_images[idx])
        clean_img = Image.open(self.clean_images[idx])
        
        noisy_img = noisy_img.convert('RGB')
        clean_img = clean_img.convert('RGB')
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img

def train_model(dataset_path, epochs=3, batch_size=16, learning_rate=0.001):
    # Explicit GPU configuration
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()  # Clear GPU memory
    else:
        device = torch.device('cpu')
        print("No GPU available. Using CPU.")

    # Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Create dataset and dataloader
    dataset = SIDDDataset(dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True  # Improve GPU data transfer
    )
    
    # Model initialization with multi-GPU support
    model = CustomDenoiseCNN()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        print(f"\nEpoch [{epoch+1}/{epochs}]:")
        for batch_idx, (noisy_images, clean_images) in enumerate(dataloader):
            noisy_images = noisy_images.to(device)
            clean_images = clean_images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            denoised_images = model(noisy_images)
            
            # Compute loss
            loss = criterion(denoised_images, clean_images)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Print detailed batch information
            print(f"  Batch [{batch_idx+1}/{len(dataloader)}], "
                  f"Batch Loss: {loss.item():.4f}, "
                  f"Running Avg Loss: {total_loss/(batch_idx+1):.4f}")
        
        scheduler.step()
        print(f"Epoch Average Loss: {total_loss/len(dataloader):.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/sidd_denoiser.pth')
    print("\nModel saved successfully!")
    
    return model

if __name__ == '__main__':
    # Replace with your actual dataset path
    dataset_path = r'C:\Users\borut\Desktop\New folder (2)\Data'
    train_model(dataset_path)