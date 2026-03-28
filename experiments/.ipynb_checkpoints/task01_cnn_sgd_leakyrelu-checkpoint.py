"""experiments/task01_cnn_sgd_leakyrelu.py — CNN SGD + LeakyReLU on CIFAR-10"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, config
from data.cifar10_loader import get_cifar10_loaders
from models.simple_cnn   import SimpleCNN
from training.trainer    import train_model

EXPERIMENT_NAME = "Task01_CNN_SGD_LeakyReLU"

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    train_loader, test_loader = get_cifar10_loaders(image_size=32, batch_size=config.BATCH_SIZE)
    model = SimpleCNN(num_classes=10, activation="leakyrelu")
    best_acc = train_model(model, train_loader, test_loader,
                           {**config.CNN_SGD_CONFIG, "device": config.DEVICE},
                           EXPERIMENT_NAME, config.TENSORBOARD_LOG_DIR)
    print(f"[Result] {EXPERIMENT_NAME}  Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
