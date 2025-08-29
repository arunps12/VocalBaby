# scripts/train.py
import argparse
from vocalbaby.pipeline import train_model, compute_metrics
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset, SPLIT_COUNTS
from sklearn.metrics import accuracy_score, recall_score
import numpy as np


def main(args):
    df = load_metadata()
    dataset = create_child_disjunct_dataset(df, split_counts=SPLIT_COUNTS)
    train_df = dataset["train"].to_pandas()
    eval_df = dataset["validation"].to_pandas()

    train_model(
        train_df=train_df,
        eval_df=eval_df,
        base_model_path=args.model,
        output_dir=args.output_dir,
        use_class_weights=args.class_weights,
        use_balancing=args.balancing,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        compute_metrics=compute_metrics,
        push_to_hub=args.push_to_hub,
        mode=args.mode, 
        prosody_model=args.prosody_model
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--output_dir", type=str, default="results/wav2vec2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--class_weights", action="store_true")
    parser.add_argument("--balancing", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to Hugging Face Hub")
    parser.add_argument("--mode", type=str, choices=["joint", "prosody", "audio"], default="joint")
    parser.add_argument("--prosody_model", type=str, choices=["cnn", "lstm"], default=None,help="Prosody model to use. If not provided, default None.")
    args = parser.parse_args()
    main(args)
