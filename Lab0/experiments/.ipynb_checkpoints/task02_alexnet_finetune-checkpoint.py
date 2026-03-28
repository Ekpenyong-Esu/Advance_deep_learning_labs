"""experiments/task02_alexnet_finetune.py — AlexNet Fine-Tuning on CIFAR-10"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, config
from data.cifar10_loader  import get_cifar10_loaders
from models.alexnet_model import get_alexnet_finetune
from training.trainer     import train_model

EXPERIMENT_NAME = "Task02_AlexNet_FineTuning"

def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    train_loader, test_loader = get_cifar10_loaders(image_size=224,
                                                    batch_size=config.ALEXNET_FINETUNE_CONFIG["batch_size"])
    model = get_alexnet_finetune(num_classes=10, pretrained=True)
    best_acc = train_model(model, train_loader, test_loader,
                           {**config.ALEXNET_FINETUNE_CONFIG, "device": config.DEVICE},
                           EXPERIMENT_NAME, config.TENSORBOARD_LOG_DIR)
    print(f"[Result] {EXPERIMENT_NAME}  Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
