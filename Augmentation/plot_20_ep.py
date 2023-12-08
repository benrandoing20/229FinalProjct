import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('Augmentation/loss_data_csv/loss_data_all_20_epochs.csv')

# Filter data for each model
data_model_1 = data[data['model'] == 'A0_B0_C0_D0']
data_model_2 = data[data['model'] == 'A4_B70_C2_D3']

# Plotting
plt.figure(figsize=(15, 6))

# Plot for model A0_B0_C0_D0
plt.plot(data_model_1['epoch'], data_model_1['train_loss'], label='Model A0_B0_C0_D0 - Train Loss', color='blue')
plt.plot(data_model_1['epoch'], data_model_1['val_loss'], label='Model A0_B0_C0_D0 - Validation Loss', color='cyan')

# Plot for model A4_B70_C2_D3
plt.plot(data_model_2['epoch'], data_model_2['train_loss'], label='Model A4_B70_C2_D3 - Train Loss', color='red')
plt.plot(data_model_2['epoch'], data_model_2['val_loss'], label='Model A4_B70_C2_D3 - Validation Loss', color='orange')

# Titles and labels
plt.title('Training and Validation Loss Across Epochs for Both Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 0.5)
plt.legend()
plt.grid(True)

# Show plot
plt.show()
