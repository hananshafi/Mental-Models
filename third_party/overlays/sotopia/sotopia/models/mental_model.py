# models/mental_model.py
from __future__ import annotations
from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

class MentalModel(nn.Module):
    """
    Unified Mental Model with Mixture-of-Experts (MoE) decoding.

    Structure:
      1. Encode History h -> (c_logits, mu, logvar)
      2. Sample z ~ N(mu, std)
      3. Gate c = softmax(c_logits) OR gumbel_softmax(c_logits)
      4. Decode v_hat = Sum(gate_k * Expert_k(z))
    """

    def __init__(
        self,
        encoder_name: str = "distilbert-base-uncased",
        num_types: int = 8,
        z_dim: int = 16,
        v_dim: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size

        self.num_types = num_types
        self.z_dim = z_dim
        self.v_dim = v_dim

        # Heads
        self.dropout = nn.Dropout(dropout)
        self.c_head = nn.Linear(hidden, num_types)
        self.mu_head = nn.Linear(hidden, z_dim)
        self.logvar_head = nn.Linear(hidden, z_dim)

        # MoE: One expert per archetype mapping Nuance (z) -> Outcomes (v)
        # We use a ModuleList for clarity, but could use BatchedLinear for speed if num_types > 50
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(z_dim, z_dim * 2),
                nn.ReLU(),
                nn.Linear(z_dim * 2, v_dim)
            )
            for _ in range(num_types)
        ])

    def encode_history(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token and apply dropout for robustness
        return self.dropout(out.last_hidden_state[:, 0])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        c_override: Optional[torch.Tensor] = None,  # (B,) labels for forced teaching
        use_hard_c: bool = False,
        gumbel_temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. Encode
        h = self.encode_history(input_ids, attention_mask)

        # 2. Latent Distributions
        c_logits = self.c_head(h)       # (B, num_types)
        mu = self.mu_head(h)            # (B, z_dim)
        logvar = self.logvar_head(h)    # (B, z_dim)

        # 3. Reparameterization (z)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # 4. Gating Mechanism (c)
        if c_override is not None:
            # SUPERVISED MODE: Force the gate to be the label
            if c_override.ndim == 1:
                gate = F.one_hot(c_override, num_classes=self.num_types).float()
            else:
                gate = c_override.float()
        else:
            # UNSUPERVISED / INFERENCE MODE
            if self.training and use_hard_c:
                # Gumbel-Softmax allows "hard" sampling while keeping gradients!
                gate = F.gumbel_softmax(c_logits, tau=gumbel_temperature, hard=True)
            elif use_hard_c:
                # Pure inference (non-differentiable is fine here)
                idx = torch.argmax(c_logits, dim=-1)
                gate = F.one_hot(idx, num_classes=self.num_types).float()
            else:
                # Soft Mixture
                gate = F.softmax(c_logits, dim=-1)

        # 5. Expert Execution (Vectorized)
        # We compute all experts for all items. Since num_experts=8 is small,
        # this is faster than sparse selection on GPUs due to batching efficiency.

        # expert_outputs shape: (B, num_types, v_dim)
        expert_outputs = torch.stack([expert(z) for expert in self.experts], dim=1)

        # Weighted Sum: (B, num_types, 1) * (B, num_types, v_dim) -> sum -> (B, v_dim)
        v_hat = torch.sum(expert_outputs * gate.unsqueeze(-1), dim=1)

        return c_logits, mu, logvar, v_hat

    @staticmethod
    def kl_normal(mu, logvar):
        # KL(N(mu, sigma) || N(0, 1))
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)+
