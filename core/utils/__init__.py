# core/utils/__init__.py
import torch
import matplotlib.pyplot as plt

class Utils():
    def select_cuda_device(x=None):
        if x:
            return x
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
        return device
    def plot_loss_curve(loss_values):
        plt.figure(figsize=(10, 6))
        plt.plot(loss_values, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()