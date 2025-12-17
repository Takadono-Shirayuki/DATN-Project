import json
import matplotlib.pyplot as plt
import os
import sys

HISTORY_FILE = "training_history.json"
OUTPUT_IMAGE = "training_result.png"

def plot_history():
    if not os.path.exists(HISTORY_FILE):
        print(f"Error: {HISTORY_FILE} not found. Run train.py first.")
        return

    # Load JSON data
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        print("Error: Could not decode JSON. File might be being written to. Try again.")
        return

    epochs = history['epoch']
    
    if len(epochs) == 0:
        print("JSON file is empty. Wait for the first epoch to finish.")
        return

    print(f"Plotting results for {len(epochs)} epochs...")

    plt.figure(figsize=(14, 6))

    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc', marker='o', color='green')
    plt.plot(epochs, history['val_acc'], label='Val Acc', marker='o', color='red')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (0-1)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE)
    print(f"Graph saved to '{OUTPUT_IMAGE}'")
    plt.show()

if __name__ == "__main__":
    plot_history()