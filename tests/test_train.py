from vocalbaby.pipeline import train_model
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset
def main():
    # Load metadata and split
    df = load_metadata()
    dataset = create_child_disjunct_dataset(df)

    # Run a minimal training job
    train_model(
        train_df=dataset["train"].to_pandas().head(1000),  
        eval_df=dataset["validation"].to_pandas().head(20),
        base_model_path="facebook/wav2vec2-base",
        output_dir="vocalbaby-test-run",
        use_class_weights=True,
        use_balancing=True,
        epochs=1,
        batch_size=2
    )

if __name__ == "__main__":
    main()