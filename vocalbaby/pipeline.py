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


def preprocess_example(example, processor, max_length=16000):
    waveform = load_and_preprocess_audio(example['path'])
    if example.get('augmented', False):
        waveform = augment(samples=waveform, sample_rate=16000)
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    prosody_signal = prosody_to_sinusoid(pitch, energy)
    #padded = apply_center_padding(prosody_signal, target_len=max_length)
    inputs = processor(prosody_signal, sampling_rate=16000, return_tensors="pt", padding=False)
    #prosody = extract_prosodic_features(padded, sr=16000)
    #prosody_signal = prosody_to_sinusoid(prosody)
    return {
        'input_values': inputs['input_values'][0],
        'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_values'][0])),
        'label': LABEL2ID[example['label']],
        'prosody': prosody_signal
    }


def train_model(train_df, eval_df, base_model_path, output_dir, use_class_weights=False, use_balancing=True, epochs=10, batch_size=8):
    # Balance training data if specified
    if use_balancing:
        train_df = balance_dataset(train_df)

    # Encode labels for reference (though we handle inside preprocess)
    #train_df = encode_labels_column(train_df)
    #eval_df = encode_labels_column(eval_df)

    model, processor = load_model_for_training(base_model_path)

    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    # Apply waveform + attention_mask + label mapping
    train_dataset = train_dataset.map(lambda ex: preprocess_example(ex, processor))
    eval_dataset = eval_dataset.map(lambda ex: preprocess_example(ex, processor))

    if use_class_weights:
        labels = train_dataset['label']
        class_weights = compute_class_weights(labels, num_classes=len(LABEL2ID))
        model.class_weights = class_weights

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        learning_rate=1e-5,
        logging_steps=10,
        push_to_hub=False,
        hub_model_id=output_dir.split("/")[-1],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor
    )

    trainer.train()
    #trainer.push_to_hub()
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
