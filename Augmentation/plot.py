import pandas as pd
import matplotlib.pyplot as plt
import os

plot_all = True

 # Load the data from the CSV file
if plot_all == True:
    data = pd.read_csv('Augmentation/loss_data_csv/loss_data_all.csv')
else:
    data = pd.read_csv('Augmentation/loss_data_csv/loss_data_all_10_epochs.csv')

# Get unique models
models = data['model'].unique()

filtered_data = data[data['model'] != 'A5_B60_C1_D2']

# Recalculate global min and max loss
global_min_loss = min(filtered_data['train_loss'].min(), filtered_data['val_loss'].min())
global_max_loss = max(filtered_data['train_loss'].max(), filtered_data['val_loss'].max())

if plot_all == True:
    # Calculate specific min and max loss for 'A5_B60_C1_D2'
    model_specific_data = data[data['model'] == 'A5_B60_C1_D2']
    model_specific_min_loss = min(model_specific_data['train_loss'].min(), model_specific_data['val_loss'].min())
    model_specific_max_loss = max(model_specific_data['train_loss'].max(), model_specific_data['val_loss'].max())

# Create a directory to save the plots
plot_dir = "model_plots"
os.makedirs(plot_dir, exist_ok=True)

# Adjust font size for better visibility
plt.rcParams.update({'font.size': 10})

# Plotting each model's graph with adjusted scale
for model in models:
    model_data = data[data['model'] == model]

    # Set min and max loss depending on the model
    min_loss = model_specific_min_loss if model == 'A5_B60_C1_D2' else global_min_loss
    max_loss = model_specific_max_loss if model == 'A5_B60_C1_D2' else global_max_loss

    # Adjust figure size to reduce height
    plt.figure(figsize=(5, 3))  # Reduced height from 4 to 2.5
    plt.plot(model_data['epoch'], model_data['train_loss'], label='Train Loss')
    plt.plot(model_data['epoch'], model_data['val_loss'], label='Validation Loss')
    plt.title(f'Model {model}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(min_loss, max_loss)
    plt.xticks(model_data['epoch'])
    plt.legend()
    plt.grid(True)

    plt.tight_layout()  # Adjust layout to prevent label cropping

    # Save the plot
    plot_path = os.path.join(plot_dir, f"{model}_loss_plot{'_10_epochs' if not plot_all else ''}.png")
    plt.savefig(plot_path)
    plt.close()

print("Plots saved in the 'model_plots' directory.")
