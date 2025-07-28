import os
import torch
from transformers import TrainingArguments, Trainer
import numpy as np
from datasets import Dataset
from vocalbaby.model import load_model_for_training
from vocalbaby.utils import encode_labels_column, compute_class_weights, balance_dataset
from vocalbaby.preprocess import load_and_preprocess_audio, apply_center_padding
from vocalbaby.feature import extract_prosodic_features, prosody_to_sinusoid
from vocalbaby.labels import ID2LABEL, LABEL2ID

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
        waveform_padded = apply_center_padding(waveform, 16000)
    # extract both features
    inputs = processor(waveform_padded, sampling_rate=16000, return_tensors="pt", padding=True)
    pitch, energy = extract_prosodic_features(waveform, sr=16000)
    prosody_signal = prosody_to_sinusoid(pitch, energy)
    #waveform = apply_center_padding(waveform, max_length)
    prosody_signal = apply_center_padding(prosody_signal, max_length)

    return {
        'input_values': inputs['input_values'][0],
        'attention_mask': inputs.get('attention_mask', torch.ones_like(inputs['input_values'][0])),
        'label': LABEL2ID[example['label']],
        'features': prosody_signal
    }


def train_model(train_df, eval_df, base_model_path, output_dir,
                use_class_weights=True, use_balancing=True,
                learning_rate=1e-5, epochs=10, batch_size=8,lr_scheduler_type="linear",
                warmup_ratio=0.1,
                compute_metrics=None, push_to_hub=False,
                stage_training=False):

    from transformers import TrainerCallback

    if use_balancing:
        train_df = balance_dataset(train_df)

    model, processor = load_model_for_training(base_model_path)

    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)

    train_dataset = train_dataset.map(lambda ex: preprocess_example(ex, processor))
    eval_dataset = eval_dataset.map(lambda ex: preprocess_example(ex, processor))

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
            push_to_hub=push_to_hub,
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

    if stage_training:
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
    else:
        trainer = make_trainer(model, output_dir, num_epochs=epochs, learning_rate=learning_rate)
        trainer.train()

    if push_to_hub:
        trainer.push_to_hub()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
