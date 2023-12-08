from simple_augment import prepare_dataset, load_augmented_data
import os
import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd

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

        return image, label, img_name

def load_model(model_path):
    print(f"Loading model from {model_path}")
    model = resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, 4)  # Assuming 4 classes as in your training script
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.eval()
    return model

def test_model(model, test_loader):
    print(f"Testing model")
    correct = 0    
    total = 0
    # Adjust the last bucket to include 160
    bucket_accuracy = {i: {'correct': 0, 'total': 0} for i in range(100, 151, 10)}

    with torch.no_grad():
        print('Entering')
        for images, labels, paths in test_loader:
            print(images)
            print('In the loop')
            print(total)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # if total % 500 == 0:
            #     print(f"Tested {total} images so far...")

            # Process each image in the batch
            for path, label, prediction in zip(paths, labels, predicted):
                number = int(path.split('_')[-1].split('.')[0])
                # Adjust bucket assignment to include 160 in the last bucket
                bucket = (number // 10) * 10
                if number == 160:
                    bucket = 150
                if 100 <= bucket <= 150:
                    bucket_accuracy[bucket]['total'] += 1
                    if label == prediction:
                        bucket_accuracy[bucket]['correct'] += 1

    model_accuracy = {'overall_accuracy': 100 * correct / total}
    for bucket in range(100, 151, 10):
        if bucket_accuracy[bucket]['total'] > 0:
            acc = 100 * bucket_accuracy[bucket]['correct'] / bucket_accuracy[bucket]['total']
            model_accuracy[f'bucket_{bucket}-{bucket+9 if bucket < 150 else 160}'] = acc
        else:
            model_accuracy[f'bucket_{bucket}-{bucket+9 if bucket < 150 else 160}'] = 'N/A'

    return model_accuracy


def main():
    # Load and prepare the new dataset
    csv_filename = 'dataset.csv'
    save_dir = 'resized_images'  # Directory for resized images from new dataset
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    df = prepare_dataset(csv_filename, save_dir)

    sampled_df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df

    print('Creating the test dataset and loader for sampled images.')
    test_dataset = CustomDataset(sampled_df, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)

    model_paths = [
        'Augmentation/saved_models/model_A0_B0_C0_D0.pth',
        'Augmentation/saved_models/model_A3_B75_C2_D3.pth',
        'Augmentation/saved_models/model_A4_B70_C2_D3.pth',
        'Augmentation/saved_models/model_A5_B60_C1_D2\.pth',
        'Augmentation/saved_models/model_A6_B80_C1_D3.pth',
        'Augmentation/saved_models/model_A7_B50_C1_D1.pth'
    ]

    # Dictionary to store image count per bucket
    bucket_counts = {i: 0 for i in range(100, 151, 10)}

    # Store results for each model
    results = []

    for model_path in model_paths:
        model = load_model(model_path)
        accuracy_results = test_model(model, test_loader)
        accuracy_results['model'] = os.path.basename(model_path).split('.')[0]
        print(f"Model {model_path}: Test Accuracy = {accuracy_results}%")
        
        # Add bucket counts to results
        for bucket in range(100, 151, 10):
            accuracy_results[f'count_bucket_{bucket}-{bucket+9 if bucket < 150 else 160}'] = bucket_counts[bucket]

        results.append(accuracy_results)

    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('model_accuracy_results.csv', index=False)
    print('Saved model accuracy results to "model_accuracy_results.csv"')

if __name__ == "__main__":
    main()
