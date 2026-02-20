"""Model definitions for OOD detection."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class IntentClassifier(nn.Module):
    """BERT-based intent classifier with penultimate-layer access."""

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False,
    ):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled = outputs.last_hidden_state[:, 0, :]
        features = self.dropout(pooled)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits

    def get_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Return penultimate-layer [CLS] embeddings (no dropout)."""
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state[:, 0, :]


class MCDropoutClassifier(IntentClassifier):
    """Same as IntentClassifier but dropout stays active at inference."""

    def forward_mc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        n_passes: int = 20,
    ) -> torch.Tensor:
        """Run n stochastic forward passes and return mean softmax."""
        self.train()  # keep dropout active
        probs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = super().forward(input_ids, attention_mask)
                probs.append(torch.softmax(logits, dim=-1))
        self.eval()
        return torch.stack(probs).mean(0)
