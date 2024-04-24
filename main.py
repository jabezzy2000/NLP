import torch
import torchvision
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
import nlp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os
from torchvision.models import resnet18, ResNet18_Weights



def get_images_in_directory(directory):
    """Get the first image from the directory."""
    lst = []
    for entry in os.listdir(directory):
        if entry.lower().endswith(('.png', '.jpg', '.jpeg')):
            #return os.path.join(directory, entry)
            lst.append(os.path.join(directory, entry))
    return lst if lst else None  # Return None if no image file is found


class XrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.xray_frame = pd.read_csv(csv_file)  
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.xray_frame)

    def __getitem__(self, idx):
        directory_path = os.path.join(self.root_dir, self.xray_frame.iloc[idx]['path'])
        img_path = get_images_in_directory(directory_path)[0]
        if img_path is None:
            print(f"No image found in directory {directory_path}")
            return None  # Return None when no image is found

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        label_names = self.xray_frame.columns[1:-1] 
        print("Label names:", label_names)
        labels = self.xray_frame.iloc[idx, 1:-1].values.astype('float')
        print("Label values for the current sample:", labels)
        labels = torch.tensor(labels)
        # print(type(labels))
        if not isinstance(image, torch.Tensor) or not isinstance(labels,torch.Tensor):
            print("Error: is not instance tensor")
            print(f"No valid image for index {idx}, returning dummy data")
            image = torch.zeros(3, 224, 224) 
            labels = torch.zeros(5) 
            return {'image': image, 'labels': labels}
        return {'image': image, 'labels': labels}


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#implementing data augmentation
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Flips the image horizontally with a 50% chance
    transforms.RandomRotation(10),      # Rotates the image between -10 and 10 degrees
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) # Loading pretrained ResNet18
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 5)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_path = 'model.pth'
#checking if fine-tuned model exists to retrain else fine-tuning pretrained model
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Loaded fine-tuned model.")
else:
    print("Loaded pretrained model.")



dataset = XrayDataset(csv_file='new_file.csv', root_dir='', transform=augmentation_transforms)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

def train_model():
    model.train()
    for epoch in range(10):
        for i, data in enumerate(dataloader):
            if data is None:
                continue  # Skip this batch if data is not loaded properly
            images, labels = data['image'], data['labels']
            # print(f"Images type: {type(images)}, Images shape: {images.shape}")
            if isinstance(images, torch.Tensor):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
                print(f"Batch {i} processed, Loss: {loss.item()}")
            else:
                print("Error: Images are not a tensor, skipping batch.")
        print(f'Epoch [{epoch+1}/10], Finished')


if __name__ == "__main__":
    train_model()
    torch.save(model.state_dict(), 'model.pth')



