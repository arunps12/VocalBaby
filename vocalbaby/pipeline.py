import os
import torch
import numpy as np
import json
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import accuracy_score, recall_score
from vocalbaby.model import load_model_for_training
from vocalbaby.utils import encode_labels_column, compute_class_weights, balance_dataset
from vocalbaby.preprocess import load_and_preprocess_audio, apply_center_padding
from vocalbaby.feature import extract_prosodic_features, prosody_to_sinusoid
from vocalbaby.labels import ID2LABEL, LABEL2ID


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    acc = accuracy_score(labels, preds)
    uar = recall_score(labels, preds, average='macro')
    return {
        "accuracy": acc,
        "uar": uar
    }


def classify_audio(audio_path, model_repo, max_length=16000):
    model, processor = load_model_for_training(model_repo)
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
        from vocalbaby.utils import CLASS_AUGMENTATIONS
        label = example['label']
        augment = CLASS_AUGMENTATIONS[label]
        #idx = int(example.get("augment_type", 0)) % len(augmentations)
        #waveform = augmentations[idx](samples=waveform, sample_rate=16000)
        waveform = augment(samples=waveform, sample_rate=16000)
    waveform_padded, attn_mask = apply_center_padding(waveform, 16000)
    # extract both features
    inputs = processor(waveform_padded, sampling_rate=16000, return_tensors="pt", padding=False)
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    prosody_signal = prosody_to_sinusoid(pitch, energy)
    #waveform = apply_center_padding(waveform, max_length)
    prosody_signal, _ = apply_center_padding(prosody_signal, max_length)

    return {
        'input_values': inputs['input_values'][0],
        'attention_mask': torch.tensor(attn_mask, dtype=torch.long),#inputs.get('attention_mask', torch.ones_like(inputs['input_values'][0])),
        'labels': LABEL2ID[example['label']],
        'prosody_signal': prosody_signal
    }


def train_model(train_df, eval_df, base_model_path, output_dir,
                use_class_weights=True, use_balancing=True,
                learning_rate=1e-5, epochs=10, batch_size=8,lr_scheduler_type="linear",
                warmup_ratio=0.1,
                compute_metrics=None,
                mode="joint", prosody_model="cnn"):

    from transformers import TrainerCallback

    if use_balancing:
        train_df = balance_dataset(train_df)

    model, processor = load_model_for_training(base_model_path, mode=mode, prosody_model=prosody_model)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    train_dataset = train_dataset.map(lambda ex: preprocess_example(ex, processor))
    eval_dataset = eval_dataset.map(lambda ex: preprocess_example(ex, processor))
    train_dataset = train_dataset.remove_columns(["label"])
    eval_dataset = eval_dataset.remove_columns(["label"])

    if use_class_weights:
        labels = train_dataset['label']
        class_weights = compute_class_weights(labels, num_classes=len(LABEL2ID))
        model.class_weights = class_weights

    def make_trainer(model_to_use, output_path, num_epochs, learning_rate):
        args = TrainingArguments(
            output_dir=output_path,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_dir=os.path.join(output_path, "logs"),
            logging_strategy="epoch",
            logging_steps=10,
            hub_model_id=output_path.split("/")[-1],
            fp16=torch.cuda.is_available(),
            report_to="none"
        )
        return Trainer(
            model=model_to_use,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor,
            compute_metrics=compute_metrics
        )

    if mode == "joint":
        print("[Stage 1] Training Prosody Branch Only")
        model.wav2vec2.eval()
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
        trainer_stage1 = make_trainer(model, output_dir + "_stage1", num_epochs=epochs*5, learning_rate= learning_rate * 10)
        trainer_stage1.train()

        print("[Stage 2] Fine-tuning Full Model")
        for param in model.wav2vec2.parameters():
            param.requires_grad = True
        trainer_stage2 = make_trainer(model, output_dir, num_epochs=epochs, learning_rate=learning_rate)
        trainer_stage2.train()
        trainer_stage2.save_model(output_dir)
    else:
        trainer = make_trainer(model, output_dir, num_epochs=epochs, learning_rate=learning_rate)
        trainer.train()
        trainer.save_model(output_dir) 

    # Saving logic
    #if mode == "audio":
        #model.save_pretrained(output_dir)
        #if push_to_hub:
            #trainer.push_to_hub()
    #else:
    #trainer.save_model(output_dir)         # saves best model (after load_best_model_at_end=True)
    processor.save_pretrained(output_dir) # save tokenizer / processor config
    #torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    #processor.save_pretrained(output_dir)

    # Save model config for custom loading
    config_dict = {
        "model_type": "DualBranchProsodyModel",
        "base_model": base_model_path,
        "mode": mode,
        "prosody_model": prosody_model,
        "num_labels": len(LABEL2ID),
        "fusion_dim": 768
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

