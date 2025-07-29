import os
import numpy as np
import torch
import pytest
from transformers import EvalPrediction
from vocalbaby.pipeline import train_model, compute_metrics
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset, SPLIT_COUNTS
from vocalbaby.model import DualBranchProsodyModel


@pytest.mark.parametrize("mode", ["audio", "prosody", "joint"])
def test_train_model_runs(mode, tmp_path):
    df = load_metadata()
    dataset = create_child_disjunct_dataset(df, split_counts=SPLIT_COUNTS)

    output_dir = tmp_path / f"test-output-{mode}"
    output_dir.mkdir()

    train_df = dataset["train"].to_pandas().sample(n=100)
    eval_df = dataset["validation"].to_pandas().sample(n=10)

    train_model(
        train_df=train_df,
        eval_df=eval_df,
        base_model_path="facebook/wav2vec2-base",
        output_dir=str(output_dir),
        use_class_weights=False,
        use_balancing=False,
        learning_rate=5e-4,
        epochs=1,
        batch_size=2,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        compute_metrics=compute_metrics,
        mode=mode,  # audio, prosody, or joint
        prosody_model="lstm" # cnn or lstm
    )

    print(f"\nFiles saved under output_dir ({mode} mode):")
    for f in output_dir.rglob("*"):
        print(f.relative_to(output_dir))

    # Assert model checkpoint was saved
    def has_model_checkpoint(path):
        return any(path.rglob("model.safetensors")) or any(path.rglob("pytorch_model.bin"))

    assert has_model_checkpoint(output_dir), f"No model checkpoint found in {output_dir}"

    # Check processor and config files
    config_files = ["config.json", "preprocessor_config.json"]
    for fname in config_files:
        assert any(f.name == fname for f in output_dir.rglob("*")), f"{fname} not found"


def test_compute_metrics():
    # Simulate predictions and labels
    logits = np.array([[0.1, 0.9, 0.0],
                       [0.8, 0.1, 0.1],
                       [0.2, 0.2, 0.6]])
    labels = np.array([1, 0, 2])

    pred = EvalPrediction(predictions=logits, label_ids=labels)
    result = compute_metrics(pred)
    print("Metrics:", result)
    assert "accuracy" in result
    assert "uar" in result
    assert np.isclose(result["accuracy"], 1.0)
    assert np.isclose(result["uar"], 1.0)


def test_model_forward_shapes():
    model = DualBranchProsodyModel("facebook/wav2vec2-base", mode="joint", prosody_model="cnn")
    B, T = 2, 16000
    prosody_len = 16000

    input_values = torch.randn(B, T)
    attention_mask = torch.ones(B, T)
    prosody_signal = torch.randn(B, prosody_len)
    labels = torch.tensor([0, 1])

    out = model(input_values=input_values, attention_mask=attention_mask,
                prosody_signal=prosody_signal, labels=labels)
    print("\nModel output:", out)
    assert "logits" in out
    assert out["logits"].shape == (B, 5)
    assert "loss" in out
    assert out["loss"].requires_grad is True
