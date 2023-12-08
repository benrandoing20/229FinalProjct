import gc
import os
import sys
import itertools
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torch.optim.lr_scheduler
from torchvision.transforms import ToPILImage
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        image = Image.open(img_name).convert('RGB')
        label = int(self.dataframe.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        # Ensure the image is the correct size
        if image.size() != (3, 224, 224):
            raise ValueError(f"Image size mismatch: expected (3, 224, 224), got {image.size()} for image {img_name}")

        return image, label

# Data Augmentation
def augment_data_and_save(df, class_id, num_augments, save_dir):
    # Create a specific folder for each class within the main augmented directory
    augmented_class_dir = os.path.join(save_dir, f"augmented_{class_id}")
    if not os.path.exists(augmented_class_dir):
        os.makedirs(augmented_class_dir)
    
    for idx, row in df[df['label'] == class_id].iterrows():
        image = Image.open(row['path'])
        for i in range(num_augments):
            transform = transforms.Compose([
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
            augmented_image = transform(image)
            augmented_image_pil = ToPILImage()(augmented_image)
            augmented_image_path = os.path.join(augmented_class_dir, f"{class_id}_{idx}_{i}.jpg")
            augmented_image_pil.save(augmented_image_path)

def resize_and_save(image_path, output_size, save_dir):
    image = Image.open(image_path).convert('RGB')
    resize_transform = transforms.Resize(output_size)
    resized_image = resize_transform(image)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    resized_image.save(save_path)
    return save_path

# Load dataset
def prepare_dataset(csv_filename, save_dir):
    df = pd.read_csv(csv_filename)

    # Check if the save directory is empty
    if not os.listdir(save_dir):
        print(f"Populating '{save_dir}' with resized images.")
        resized_image_paths = []
        for _, row in df.iterrows():
            new_path = resize_and_save(row['path'], (224, 224), save_dir)
            resized_image_paths.append([new_path, row['label']])
        return pd.DataFrame(resized_image_paths, columns=['path', 'label'])
    else:
        print(f"'{save_dir}' already contains images. Skipping resizing.")
        resized_image_paths = [[os.path.join(save_dir, os.path.basename(row['path'])), row['label']] for _, row in df.iterrows()]
        return pd.DataFrame(resized_image_paths, columns=['path', 'label'])


# Load Augmented data
def load_augmented_data(augmented_dir):
    augmented_data = []

    # Check and iterate through each subfolder in the augmented_dir
    for subfolder in os.listdir(augmented_dir):
        subfolder_path = os.path.join(augmented_dir, subfolder)
        if os.path.isdir(subfolder_path):  # Ensure it is a directory
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpg'):
                    class_id = int(filename.split('_')[0])  # Extracting class_id from filename
                    img_path = os.path.join(subfolder_path, filename)
                    augmented_data.append([img_path, class_id])

    return augmented_data


# Model Training Function
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs):
    epoch_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Print progress every 50 batches
            if (i+1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Batch {i+1}/{len(train_loader)}")

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Print progress every 50 batches
                if (i+1) % 40 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} - Val Batch {i+1}/{len(val_loader)}")

        avg_val_loss = val_loss / len(val_loader)

        epoch_losses.append((avg_train_loss, avg_val_loss))
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        sys.stdout.flush()

    return epoch_losses


# Evaluate Model
def evaluate_model(test_loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def main():

    print('Start.')

    # Load and Prepare Dataset
    csv_filename = 'original_dataset.csv'
    save_dir = 'resized_images'  # Directory to save resized images
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print('Preparing the dataset...')
    df = prepare_dataset(csv_filename, save_dir)

    # Split Original Dataset into train, validation, and test sets
    train_df, test_val_df = train_test_split(df, test_size=0.4, random_state=42)
    val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

    # Print sizes of the datasets before augmentation
    print(f"Size of original train dataset: {len(train_df)}")
    print(f"Size of original validation dataset: {len(val_df)}")
    print(f"Size of original test dataset: {len(test_df)}")

    # Define the sets of values for A, B, C, and D
    augmentation_sets = [
        {'A': 0, 'B': 0, 'C': 0, 'D': 0},
        {'A': 4, 'B': 70, 'C': 2, 'D': 3},
        {'A': 5, 'B': 60, 'C': 1, 'D': 2},
        {'A': 6, 'B': 80, 'C': 1, 'D': 3},
        {'A': 3, 'B': 75, 'C': 2, 'D': 3},
        {'A': 7, 'B': 50, 'C': 1, 'D': 1},
    ]

    best_accuracy = 0
    best_params = {}

    # Initialize a list to store loss records
    loss_data_all = []

    # Iterate over each set of values to augment the data
    for augmentation_values in augmentation_sets:
       
        # Data Augmentation
        A, B, C, D = augmentation_values.values()
        print('\n=========================\n')
        print(f"Testing with A={A}, B={B}, C={C}, D={D}")
        augmented_dir = f"augmented_images_A{A}_B{B}_C{C}_D{D}"

        total_original_images = len(df)
        total_augmented_images = 0

        # Check if the augmented directory exists
        if not os.path.exists(augmented_dir):
            os.makedirs(augmented_dir)

            # Data Augmentation
            print("Starting data augmentation...")
            for class_id, num_augments in zip([0, 1, 2, 3], [A, B, C, D]):
                augment_data_and_save(train_df, class_id, num_augments, augmented_dir)
            gc.collect()
        else:
            print(f"Using existing augmented data in {augmented_dir}")

        # Assuming df contains columns ['path', 'label'] and labels are 0, 1, 2, 3
        for class_id, num_augments in zip([0, 1, 2, 3], [A, B, C, D]):
            num_images_per_class = len(df[df['label'] == class_id])
            total_augmented_images += num_images_per_class * num_augments
        total_images = total_original_images + total_augmented_images
        print(f"Total number of images (original + augmented): {total_images}")

        # Load augmented images and combine with original training data
        print("Loading augmented images...")
        augmented_images = load_augmented_data(augmented_dir)
        augmented_train_df = pd.DataFrame(augmented_images, columns=['path', 'label'])
        combined_train_df = pd.concat([train_df, augmented_train_df])

        # Print sizes of the datasets after augmentation
        print(f"Size of combined train dataset (after augmentation): {len(combined_train_df)}")

        # Create Data Loaders for train, validation, and test sets
        print("Creating data loaders...")
        train_dataset = CustomDataset(combined_train_df, transform=transforms.ToTensor())
        val_dataset = CustomDataset(val_df, transform=transforms.ToTensor())
        test_dataset = CustomDataset(test_df, transform=transforms.ToTensor())

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=12)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)

        # Load Pre-trained ResNet18 Model
        print("Loading pre-trained ResNet18 model...")
        model = resnet18(weights=None)
        num_features = model.fc.in_features  # Modify the output layer to match the saved model's output layer
        model.fc = torch.nn.Linear(num_features, 4)  # Assuming 4 is the number of classes
        model.load_state_dict(torch.load('resnet18_more_epochs_adam_model_weights.pth'))
        model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        criterion = torch.nn.CrossEntropyLoss()

        # Define optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        # Train and Evaluate Model
        print(f"Starting training with A={A}, B={B}, C={C}, D={D}...")
        epoch_losses = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=5)
        print("Training completed.")

        accuracy = evaluate_model(test_loader, model)
        print(f"Test Accuracy with A={A}, B={B}, C={C}, D={D}: {accuracy}%")

        loss_data = []
        for epoch, (train_loss, val_loss) in enumerate(epoch_losses, 1):
            loss_data.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_accuracy': accuracy
            })
            loss_data_all.append({
                'model': f"A{A}_B{B}_C{C}_D{D}",
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'test_accuracy': accuracy
            })

        # Save the loss data for the current augmentation set to a CSV file
        loss_df = pd.DataFrame(loss_data)
        augmentation_set_name = f"A{A}_B{B}_C{C}_D{D}"
        csv_filename = f'loss_data_{augmentation_set_name}.csv'
        loss_df.to_csv(csv_filename, index=False)
        print(f"Loss data for {augmentation_set_name} saved to '{csv_filename}'")

        # Save the model
        model_save_dir = 'saved_models'
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        model_save_dir = os.path.join('Augmentation', model_save_dir)
        model_filename = os.path.join(model_save_dir, f'model_A{A}_B{B}_C{C}_D{D}.pth')
        torch.save(model.state_dict(), model_filename)
        print(f"Model for A={A}, B={B}, C={C}, D={D} saved to '{model_filename}'")

        # Update best accuracy and parameters if current combination is better
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = augmentation_values

    loss_all_df = pd.DataFrame(loss_data_all)
    csv_filename = f'loss_data_all.csv'
    loss_all_df.to_csv(csv_filename, index=False)
    print(f"Loss data for all the models saved to '{csv_filename}'")

    print(f"Best Accuracy: {best_accuracy}%")
    print(f"Best Parameters: {best_params}")
    

if __name__ == "__main__":
    main()