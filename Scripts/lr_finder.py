import os
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from datasets import Dataset
from vocalbaby.pipeline import preprocess_example
from vocalbaby.utils import load_metadata, create_child_disjunct_dataset, balance_dataset, SPLIT_COUNTS
from vocalbaby.model import load_model_for_training


def find_learning_rate(model, processor, train_dataset, eval_dataset,
                       output_dir="lr-finder-output", min_lr=1e-7, max_lr=1e-2,
                       num_iters=50, batch_size=8):
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_iters)
    losses = []

    for i, lr in enumerate(lrs):
        print(f"[{i+1}/{num_iters}] Trying LR={lr:.2e}")

        args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            num_train_epochs=1,
            max_steps=10,
            logging_steps=5,
            save_steps=1000,
            eval_steps=10,
            eval_strategy="steps",
            save_strategy="no",
            disable_tqdm=False,
            report_to="none"
        )

        temp_model = deepcopy(model)

        trainer = Trainer(
            model=temp_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor
        )

        trainer.train()
        metrics = trainer.evaluate()
        loss = metrics.get("eval_loss", None)
        print(f"Eval loss: {loss}")
        losses.append(loss)

        del temp_model
        torch.cuda.empty_cache()

    return lrs, losses


def plot_learning_rate_curve(lrs, losses):
    plt.figure(figsize=(8, 5))
    plt.plot(lrs, losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Validation Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Load and split data
df = load_metadata()
dataset = create_child_disjunct_dataset(df, split_counts=SPLIT_COUNTS)

train_df = dataset["train"].to_pandas()
train_df = balance_dataset(train_df).sample(n=1000)
eval_df = dataset["validation"].to_pandas().sample(n=100)


# Load model + processor
base_model = "facebook/wav2vec2-base"
mode = "joint" # "prosody" #"joint"  "audio", 
prosody_model = "lstm"
model, processor = load_model_for_training(base_model, mode=mode, prosody_model=prosody_model)

# Apply preprocessing
train_dataset = Dataset.from_pandas(train_df)
eval_dataset = Dataset.from_pandas(eval_df)
train_dataset = train_dataset.map(lambda ex: preprocess_example(ex, processor))
eval_dataset = eval_dataset.map(lambda ex: preprocess_example(ex, processor))
train_dataset = train_dataset.remove_columns(["label"])
eval_dataset = eval_dataset.remove_columns(["label"])

lrs, losses = find_learning_rate(
    model=model,
    processor=processor,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    min_lr=1e-6,
    max_lr=1e-2,
    num_iters=20,
    batch_size=2
)



# Convert to numpy
lrs = np.array(lrs)
losses = np.array(losses)

# Find minimum loss index
min_idx = np.argmin(losses)
lr_min_loss = lrs[min_idx]

# Steepest slope
loss_diffs = np.diff(losses)
steepest_idx = np.argmin(loss_diffs)
lr_steepest = lrs[steepest_idx]

# Before divergence
threshold = np.median(losses) + 2 * np.std(losses)
diverging = np.where(losses > threshold)[0]
if len(diverging) > 0:
    lr_pre_diverge = lrs[max(0, diverging[0] - 1)] / 10
else:
    lr_pre_diverge = lr_min_loss

# Report
print(f"\n LR w/ Min Loss:         {lr_min_loss:.2e}")
print(f" LR w/ Steepest Slope:   {lr_steepest:.2e}")
print(f"  LR Before Divergence:  {lr_pre_diverge:.2e}")

plot_learning_rate_curve(lrs, losses)