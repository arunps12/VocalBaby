import os
import pytest
from vocalbaby.pipeline import train_model
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset

@pytest.mark.parametrize("base_model", ["facebook/wav2vec2-base"])
def test_train_model_runs(base_model, tmp_path):
    df = load_metadata()
    dataset = create_child_disjunct_dataset(df)

    output_dir = tmp_path / "test-output"
    output_dir.mkdir()

    train_model(
        train_df=dataset["train"].to_pandas().head(100),
        eval_df=dataset["validation"].to_pandas().head(10),
        base_model_path=base_model,
        output_dir=str(output_dir),
        use_class_weights=False,
        use_balancing=True,
        epochs=1,
        batch_size=2
    )
    print("\nFiles saved under output_dir:")
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
