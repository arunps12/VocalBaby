import os 
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import Dataset
from vocalbaby.model import load_model_for_inference, load_model_for_training
from vocalbaby.utils import encode_labels_column, compute_class_weights, balance_dataset, augment
from vocalbaby.preprocess import load_and_preprocess_audio, apply_center_padding
from vocalbaby.feature import extract_prosodic_features, prosody_to_sinusoid
from vocalbaby.labels import ID2LABEL, LABEL2ID



def classify_audio(audio_path, model_repo, max_length=16000):
    model, processor = load_model_for_inference(model_repo)
    waveform = load_and_preprocess_audio(audio_path)
    waveform = apply_center_padding(waveform, max_length)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()

    return {
        "label": ID2LABEL[pred_id],
        "probs": probs.squeeze().tolist()
    }


def preprocess_example(example, processor,  max_length=16000, use_raw_audio=False):
    waveform = load_and_preprocess_audio(example['path'])
    if example.get('augmented', False):
        waveform = augment(samples=waveform, sample_rate=16000)
    #padded = apply_center_padding(prosody_signal, target_len=max_length)
    if use_raw_audio:
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=False)
        features = waveform
    else:
        pitch, energy = extract_prosodic_features(waveform, sr=16000)
        prosody_signal = prosody_to_sinusoid(pitch, energy)
        inputs = processor(prosody_signal, sampling_rate=16000, return_tensors="pt", padding=False)
        features = prosody_signal

    return {
        'input_values': inputs['input_values'][0],
        'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_values'][0])),
        'label': LABEL2ID[example['label']],
        'features': features
    }


import os
import torch
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from vocalbaby.utils import compute_class_weights, balance_dataset
from vocalbaby.preprocess import preprocess_example
from vocalbaby.model import load_model_for_training
from vocalbaby.labels import LABEL2ID

def train_model(train_df, eval_df, base_model_path, output_dir,
                use_class_weights=False, use_balancing=True,
                learning_rate=1e-5, epochs=10, batch_size=8,
                use_raw_audio=False, compute_metrics=None, push_to_hub=False):

    # Balance training data if specified
    if use_balancing:
        train_df = balance_dataset(train_df)

    # Load model and processor
    model, processor = load_model_for_training(base_model_path)

    # Convert to HF datasets and preprocess
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    train_dataset = train_dataset.map(lambda ex: preprocess_example(ex, processor, use_raw_audio=use_raw_audio))
    eval_dataset = eval_dataset.map(lambda ex: preprocess_example(ex, processor, use_raw_audio=use_raw_audio))

    # Add class weights if needed
    if use_class_weights:
        labels = train_dataset['label']
        class_weights = compute_class_weights(labels, num_classes=len(LABEL2ID))
        model.class_weights = class_weights

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_strategy="epoch",
        logging_steps=10,
        push_to_hub=push_to_hub,
        hub_model_id=output_dir.split("/")[-1],
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()

    # Push to hub only if specified
    if push_to_hub:
        trainer.push_to_hub()

    # Save locally
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
