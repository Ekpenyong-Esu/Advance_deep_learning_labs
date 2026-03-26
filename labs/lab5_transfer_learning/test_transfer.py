"""
Unit tests for Lab 5 transfer learning helpers.
Run with: python -m pytest test_transfer.py -v
"""

import torch
import pytest
from main import (
    build_model,
    freeze_backbone,
    unfreeze_all,
    count_trainable,
    topk_accuracy,
    NUM_CLASSES,
)


class TestBuildModel:
    def test_output_shape(self) -> None:
        model = build_model(NUM_CLASSES)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, NUM_CLASSES)

    def test_fc_layer_replaced(self) -> None:
        model = build_model(num_classes=5)
        assert model.fc.out_features == 5


class TestFreezeUnfreeze:
    def test_freeze_reduces_trainable(self) -> None:
        model = build_model()
        total = count_trainable(model)
        freeze_backbone(model)
        frozen = count_trainable(model)
        assert frozen < total

    def test_freeze_only_fc_trainable(self) -> None:
        model = build_model()
        freeze_backbone(model)
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                assert param.requires_grad
            else:
                assert not param.requires_grad

    def test_unfreeze_all(self) -> None:
        model = build_model()
        freeze_backbone(model)
        unfreeze_all(model)
        for param in model.parameters():
            assert param.requires_grad


class TestTopKAccuracy:
    def test_perfect_predictions(self) -> None:
        output = torch.eye(10)          # softmax-like: class i scores highest for sample i
        targets = torch.arange(10)
        acc1, acc5 = topk_accuracy(output, targets, topk=(1, 5))
        assert abs(acc1 - 1.0) < 1e-5
        assert abs(acc5 - 1.0) < 1e-5

    def test_zero_predictions(self) -> None:
        output = -torch.eye(10)         # class i scores *lowest* for sample i
        targets = torch.arange(10)
        acc1, _ = topk_accuracy(output, targets, topk=(1, 5))
        assert acc1 < 0.5              # should be low (not always 0 due to ties)

    def test_top5_geq_top1(self) -> None:
        output = torch.randn(32, 10)
        targets = torch.randint(0, 10, (32,))
        acc1, acc5 = topk_accuracy(output, targets, topk=(1, 5))
        assert acc5 >= acc1 - 1e-5
