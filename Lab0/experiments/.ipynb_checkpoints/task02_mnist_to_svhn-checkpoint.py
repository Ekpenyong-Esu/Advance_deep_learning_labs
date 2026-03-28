"""experiments/task02_mnist_to_svhn.py — Transfer Learning: MNIST -> SVHN"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, config
from data.mnist_loader import get_mnist_loaders
from data.svhn_loader  import get_svhn_loaders_grayscale
from models.mnist_cnn  import MnistCNN
from training.trainer  import train_model
from utils.helpers     import save_checkpoint, load_checkpoint

MNIST_CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "mnist_cnn.pth")


def stage1_train_mnist():
    print("\n" + "="*62 + "\n  STAGE 1 — Train CNN on MNIST\n" + "="*62)
    train_loader, test_loader = get_mnist_loaders(batch_size=config.BATCH_SIZE)
    model = MnistCNN(num_classes=10, input_size=28)
    best_acc = train_model(model, train_loader, test_loader,
                           {**config.MNIST_CONFIG, "device": config.DEVICE},
                           "Task02_MNIST_Training", config.TENSORBOARD_LOG_DIR)
    save_checkpoint(model, MNIST_CHECKPOINT)
    print(f"[Stage 1] MNIST Test Accuracy: {best_acc:.2f}%")
    return best_acc


def stage2_transfer_svhn():
    print("\n" + "="*62 + "\n  STAGE 2 — Transfer CNN from MNIST to SVHN\n" + "="*62)
    use_extra = config.SVHN_TRANSFER_CONFIG.get("use_extra_data", False)
    train_loader, test_loader = get_svhn_loaders_grayscale(batch_size=config.BATCH_SIZE, use_extra=use_extra)
    model = MnistCNN(num_classes=10, input_size=28)
    model = load_checkpoint(model, MNIST_CHECKPOINT, device=config.DEVICE)
    model.freeze_features()
    svhn_config = {**config.SVHN_TRANSFER_CONFIG, "device": config.DEVICE}
    svhn_config.pop("use_extra_data", None)
    best_acc = train_model(model, train_loader, test_loader, svhn_config,
                           "Task02_SVHN_Transfer", config.TENSORBOARD_LOG_DIR)
    print(f"[Stage 2] SVHN Test Accuracy (after transfer): {best_acc:.2f}%")
    return best_acc


def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    mnist_acc = stage1_train_mnist()
    svhn_acc  = stage2_transfer_svhn()
    print("\n" + "="*62)
    print(f"  MNIST accuracy (source)           : {mnist_acc:.2f}%")
    print(f"  SVHN  accuracy (after transfer)   : {svhn_acc:.2f}%")
    print("="*62)

if __name__ == "__main__":
    main()
