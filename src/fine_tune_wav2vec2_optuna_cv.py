import os
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, TrainingArguments, Trainer, Wav2Vec2FeatureExtractor
from transformers.trainer_callback import ProgressCallback
from datasets import Dataset, Audio
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import optuna
from optuna_pruning import OptunaPruningCallback
import librosa
import optuna.visualization as vis
import torch
from config import PROCESSED_METADATA_PATH, PROCESSED_DATA_DIR, MODELS_DIR

# Paths
metadata_path = PROCESSED_METADATA_PATH
audio_dir = PROCESSED_DATA_DIR
model_name = "facebook/wav2vec2-large-xlsr-53"  # Pretrained Wav2Vec 2.0 model
models_dir = MODELS_DIR
os.makedirs(models_dir, exist_ok=True)

# Device configuration: CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
num_folds = 3  # Number of cross-validation folds

# Load metadata
metadata = pd.read_csv(metadata_path)
metadata["audio_path"] = metadata["clip_ID"].apply(lambda x: os.path.join(audio_dir, x))
metadata["int_classes"] = metadata["BinaryClass"].map({"A": 0, "B": 1})
metadata = metadata.dropna(subset=["int_classes"])  
#metadata["int_classes"] = metadata["int_classes"].astype(int)
metadata = metadata[["audio_path", "int_classes"]]  
# Create dataset
dataset = Dataset.from_pandas(metadata)
dataset = dataset.cast_column("audio_path", Audio())

# Prepare cross-validation splits
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# define the metrics
def compute_metrics(pred):
    """Compute accuracy and F1 macro."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

# Define the objective function for Optuna
def objective(trial):
    fold_metrics = []

    # Define hyperparameters to optimize
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)  # Updated to use suggest_float
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    per_device_train_batch_size = trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16])

    for fold, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f"Processing fold {fold + 1}/{num_folds}")

        # Split dataset into train and validation sets
        train_dataset = dataset.select(train_index)
        val_dataset = dataset.select(val_index)

        # Load feature extractor and model
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            problem_type="single_label_classification"
        ).to(device)  # Move the model to the selected device

        # Preprocessing function
        def preprocess_function(examples, feature_extractor):
            audio = examples["audio_path"]["array"]
            inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            return {
                    "input_values": inputs["input_values"].squeeze(),
                    "labels": examples["int_classes"],  # Leave as is; we'll convert later
                    }


        # Apply preprocessing
        train_dataset = train_dataset.map(lambda x: preprocess_function(x, feature_extractor), remove_columns=["audio_path", "int_classes"])
        val_dataset = val_dataset.map(lambda x: preprocess_function(x, feature_extractor), remove_columns=["audio_path", "int_classes"])

        # Convert to PyTorch tensors
        train_dataset.set_format(type="torch", columns=["input_values", "labels"])
        # Verify the labels are tensors
        print(f"Labels DataType: {train_dataset[0]['input_values'].dtype}")
        print(f"Labels DataType: {train_dataset[0]['labels'].dtype}")

        # Convert to PyTorch tensors
        val_dataset.set_format(type="torch", columns=["input_values", "labels"])
        # Verify the labels are tensors
        print(f"Labels DataType: {val_dataset[0]['input_values'].dtype}")
        print(f"Labels DataType: {train_dataset[0]['labels'].dtype}")

        # Training arguments
        training_args = TrainingArguments(
            output_dir=os.path.join(models_dir, f"wav2vec2_cv_fold{fold + 1}"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=learning_rate,
            per_device_train_batch_size=per_device_train_batch_size,
            num_train_epochs=num_train_epochs,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            logging_dir=os.path.join(models_dir, f"logs/logs_fold{fold + 1}"),
            disable_tqdm=False  
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=feature_extractor,  
            compute_metrics=compute_metrics,
            callbacks=[OptunaPruningCallback(trial)],  
        )

        # Train and evaluate
        trainer.train()
        metrics = trainer.evaluate()
        fold_metrics.append(metrics["eval_loss"])

    # Return average validation loss across folds
    return sum(fold_metrics) / len(fold_metrics)



# Optuna study
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# Visualization of Optuna study
print("Best hyperparameters:", study.best_params)

# Generate and save visualizations
vis.plot_optimization_history(study).write_image("optimization_history.png")
vis.plot_param_importances(study).write_image("param_importances.png")
vis.plot_parallel_coordinate(study).write_image("parallel_coordinate.png")
print("Saved Optuna visualizations.")

# Final model training with best hyperparameters
def train_final_model(best_hyperparameters):
    train_index, val_index = list(kf.split(dataset))[0]  # Use first fold for final training
    train_dataset = dataset.select(train_index)
    val_dataset = dataset.select(val_index)

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

    # Preprocessing function
    def preprocess_function(examples, feature_extractor):
        audio = examples["audio_path"]["array"]
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        #print(f"Input values shape: {inputs['input_values'].shape}")
        #print(f"Label: {examples['int_classes']}")
        return {
                 "input_values": inputs["input_values"].squeeze(),
                 "labels": examples["int_classes"],  
                }

    # Apply preprocessing
    train_dataset = train_dataset.map(lambda x: preprocess_function(x, feature_extractor), remove_columns=["audio_path", "int_classes"])
    val_dataset = val_dataset.map(lambda x: preprocess_function(x, feature_extractor), remove_columns=["audio_path", "int_classes"])
    # Convert to PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_values", "labels"])
    # Verify the labels are tensors
    print(f"Labels DataType: {train_dataset[0]['input_values'].dtype}")
    print(f"Labels DataType: {train_dataset[0]['labels'].dtype}")
        
    # Convert to PyTorch tensors
    val_dataset.set_format(type="torch", columns=["input_values", "labels"])
    # Verify the labels are tensors
    print(f"Labels DataType: {val_dataset[0]['input_values'].dtype}")
    print(f"Labels DataType: {train_dataset[0]['labels'].dtype}")

    # Load model
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    ).to(device)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(models_dir, "wav2vec2_final"),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=best_hyperparameters["learning_rate"],
        per_device_train_batch_size=best_hyperparameters["per_device_train_batch_size"],
        num_train_epochs=best_hyperparameters["num_train_epochs"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=os.path.join(models_dir, "logs/logs_final"),
        disable_tqdm=False 
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=feature_extractor,  # Pass the feature extractor
        compute_metrics=compute_metrics,
        callbacks=[ProgressCallback()],  # Add progress bar
    )

    # Train
    trainer.train()

    # Save the final model
    model_dir = os.path.join(models_dir, "wav2vec2_best_model")
    model.save_pretrained(model_dir)
    feature_extractor.save_pretrained(model_dir)  # Save feature extractor
    print(f"Final model trained and saved in {model_dir}")

# Train the final model with the best hyperparameters
train_final_model(study.best_params)
