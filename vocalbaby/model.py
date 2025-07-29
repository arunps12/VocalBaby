import os
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DualBranchProsodyModel(nn.Module):
    def __init__(self, wav2vec2_path, fusion_dim=768, num_labels=5, mode="joint", prosody_model="cnn"):
        super().__init__()
        self.mode = mode  # 'joint', 'audio_only', 'prosody_only'
        self.prosody_model = prosody_model  # 'cnn' or 'lstm'

        # Load pretrained Wav2Vec2
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec2_path)
        self.wav2vec2.feature_extractor._freeze_parameters()

        if self.prosody_model == "cnn":
            self.prosody_net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
            self.prosody_fc = nn.Linear(128, fusion_dim)

        elif self.prosody_model == "lstm":
            self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1,
                                batch_first=True, bidirectional=True, dropout=0.3)
            self.prosody_fc = nn.Sequential(
                nn.Linear(64 * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, fusion_dim)
            )

        input_dim = fusion_dim if self.mode in ["audio_only", "prosody_only"] else self.wav2vec2.config.hidden_size + fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_labels)
        )

        self._init_weights()

    def forward(self, input_values=None, attention_mask=None, prosody_signal=None, labels=None):
        if self.mode == "audio_only":
            audio_embed = self.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            fused = audio_embed

        elif self.mode == "prosody_only":
            fused = self._encode_prosody(prosody_signal)

        else:  # joint
            audio_embed = self.wav2vec2(input_values=input_values, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
            prosody_embed = self._encode_prosody(prosody_signal)
            fused = torch.cat([audio_embed, prosody_embed], dim=-1)

        logits = self.classifier(fused)

    # Compute loss if labels are provided
        if labels is not None:
            if hasattr(self, "class_weights"):
                loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}



    def _encode_prosody(self, prosody_signal):
        if self.prosody_model == "cnn":
            prosody_signal = prosody_signal.unsqueeze(1)  # (B, 1, T)
            feat = self.prosody_net(prosody_signal)
            return self.prosody_fc(feat)
        elif self.prosody_model == "lstm":
            prosody_signal = prosody_signal.unsqueeze(-1)  # (B, T, 1)
            lstm_out, _ = self.lstm(prosody_signal)
            pooled = torch.mean(lstm_out, dim=1)
            return self.prosody_fc(pooled)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)



def load_model_for_inference(repo_id_or_path, mode="joint", prosody_model="cnn"):
    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(repo_id_or_path)
    model = DualBranchProsodyModel(repo_id_or_path, mode=mode, prosody_model=prosody_model)
    model.eval()
    return model, processor


def load_model_for_training(base_model_path, num_labels=5, mode="joint", prosody_model="cnn"):
    from transformers import Wav2Vec2Processor
    processor = Wav2Vec2Processor.from_pretrained(base_model_path)
    model = DualBranchProsodyModel(base_model_path, num_labels=num_labels, mode=mode, prosody_model=prosody_model)
    return model, processor
