import os
import numpy as np
import json
import torch
import pytest
import soundfile as sf
from pprint import pprint
from transformers import EvalPrediction
from vocalbaby.preprocess import load_and_preprocess_audio
from vocalbaby.pipeline import train_model, compute_metrics, classify_audio
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset, SPLIT_COUNTS
from vocalbaby.model import DualBranchProsodyModel, load_model_for_inference
from torchinfo import summary 

@pytest.mark.parametrize("mode", ["audio", "prosody", "joint"])
def test_train_model_runs(mode, tmp_path):
    df = load_metadata()
    dataset = create_child_disjunct_dataset(df, split_counts=SPLIT_COUNTS)

    output_dir = tmp_path / f"test-output-{mode}"
    output_dir.mkdir()

    train_df = dataset["train"].to_pandas().sample(n=100)
    eval_df = dataset["validation"].to_pandas().sample(n=10)

    # Set prosody_model only if needed
    prosody_model = "lstm" if mode in ["prosody", "joint"] else None

    print(f"\n=== Training Mode: {mode.upper()} ===")
    print(f"Prosody Model: {prosody_model if prosody_model else 'N/A'}")
    print(f"Output Directory: {output_dir}\n")

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
        mode=mode,
        prosody_model=prosody_model
    )

    print(f"\n Files saved under output_dir ({mode} mode):")
    for f in output_dir.rglob("*"):
        print(f.relative_to(output_dir))

    # Assert model checkpoint exists
    def has_model_checkpoint(path):
        return any(path.rglob("model.safetensors")) or any(path.rglob("pytorch_model.bin"))

    assert has_model_checkpoint(output_dir), f"No model checkpoint found in {output_dir}"

    # Confirm required config files are saved
    for fname in ["config.json", "preprocessor_config.json"]:
        assert any(f.name == fname for f in output_dir.rglob("*")), f"{fname} not found"

    # Load and display saved model config
    config_path = output_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_config = json.load(f)
        print(f"\n Loaded Model Config for '{mode}':")
        pprint(model_config)

    # Show model summary using dummy inputs
    model, _ = load_model_for_inference(str(output_dir), mode=mode, prosody_model=prosody_model)
    dummy_inputs = {
        "input_values": torch.randn(1, 16000),
        "attention_mask": torch.ones(1, 16000, dtype=torch.long)
    }
    if mode in ["prosody", "joint"]:
        dummy_inputs["prosody_signal"] = torch.randn(1, 16000)
    print(f"\n Model Architecture Summary ({mode.upper()} mode):")
    summary(model, input_data=dummy_inputs, col_names=["input_size", "output_size", "num_params"], depth=3)
    # Sanity check: count trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Trainable Parameters ({mode.upper()} mode): {num_params:,}")

    #  Prediction tests
    for i, row in eval_df.iterrows():
        wav_path = row["path"]
        waveform = load_and_preprocess_audio(wav_path)
        tmp_wav = tmp_path / f"eval_sample_{mode}_{i}.wav"
        sf.write(tmp_wav, waveform, samplerate=16000)

        result = classify_audio(str(tmp_wav), str(output_dir))
        print(f"[{mode}] Sample {i} => Label: {result['label']}, Probs: {result['probs']}")
        assert "label" in result
        assert "probs" in result
        assert isinstance(result["probs"], list)
        
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