"""experiments/grade5_transformers_cifar10.py — ViT-B/16 + Swin-T on CIFAR-10"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, config
from data.cifar10_loader       import get_cifar10_loaders
from models.vision_transformer import get_vit_model, get_swin_transformer
from training.trainer          import train_model


def run_transformer_experiment(model_name, model, exp_config, train_loader, test_loader):
    print(f"\n>>> Starting: {model_name}")
    best_acc = train_model(model, train_loader, test_loader, exp_config,
                           model_name, config.TENSORBOARD_LOG_DIR)
    print(f"[Result] {model_name}  Best Accuracy: {best_acc:.2f}%")
    return best_acc


def main():
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
    else:
        print("\nWARNING: No GPU. ViT and Swin are large — consider reducing epochs.\n")

    vit_batch  = config.VIT_CONFIG["batch_size"]
    swin_batch = config.SWIN_CONFIG["batch_size"]

    if vit_batch == swin_batch:
        train_loader, test_loader = get_cifar10_loaders(image_size=224, batch_size=vit_batch)
        vit_train, vit_test   = train_loader, test_loader
        swin_train, swin_test = train_loader, test_loader
    else:
        vit_train,  vit_test  = get_cifar10_loaders(image_size=224, batch_size=vit_batch)
        swin_train, swin_test = get_cifar10_loaders(image_size=224, batch_size=swin_batch)

    results = {}

    vit_model  = get_vit_model(num_classes=10)
    vit_config = {**config.VIT_CONFIG, "device": config.DEVICE}
    vit_config.pop("image_size", None)
    results["ViT-B/16"] = run_transformer_experiment(
        "Grade5_ViT_B16_CIFAR10", vit_model, vit_config, vit_train, vit_test)

    swin_model  = get_swin_transformer(num_classes=10)
    swin_config = {**config.SWIN_CONFIG, "device": config.DEVICE}
    swin_config.pop("image_size", None)
    results["Swin-T"] = run_transformer_experiment(
        "Grade5_SwinT_CIFAR10", swin_model, swin_config, swin_train, swin_test)

    print("\n" + "="*62 + "\n  Grade-5 Transformer Comparison")
    print("-"*62)
    for name, acc in results.items():
        print(f"  {name:<30} Best Acc: {acc:.2f}%")
    print("="*62)

if __name__ == "__main__":
    main()
