# models/reward_model.py
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class ThetaConditionedReward(nn.Module):
    """
    R_ψ(text, θ) -> scalar reward

    Architecture: Late Fusion
    1. Encoder(text) -> [CLS] embedding
    2. Fuse([CLS], c_one_hot, z) -> Hidden
    3. Project(Hidden) -> Scalar Reward
    """

    def __init__(
        self,
        base_name: str = "distilbert-base-uncased",
        num_types: int = 8,
        z_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        hidden = self.encoder.config.hidden_size

        self.num_types = num_types
        self.z_dim = z_dim

        # Dimensions: Text(768) + Archetype(8) + Nuance(16) = 792
        theta_dim = num_types + z_dim
        fusion_input_dim = hidden + theta_dim

        # Modern MLP Head: Linear -> GELU -> Dropout -> Linear
        self.head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden),
            nn.GELU(),              # Matches DistilBERT's internal activation
            nn.Dropout(dropout),    # Critical for preventing over-reliance on Theta
            nn.Linear(hidden, 1)    # Output scalar
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        c: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids, attention_mask: (B, L) - The dialogue history + candidate response
            c: (B,) long - The inferred social archetype index
            z: (B, z_dim) - The inferred continuous nuance vector
        Returns:
            r: (B,) - Scalar reward (higher is better)
        """
        # 1. Encode Text
        enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = enc.last_hidden_state[:, 0]  # (B, H)

        # 2. Build Theta (Social State)
        # Ensure c is on same device as input_ids
        one_hot_c = torch.nn.functional.one_hot(
            c, num_classes=self.num_types
        ).to(dtype=cls_emb.dtype) # match float16/32 of encoder

        # Concatenate: [BERT_CLS, OneHot_C, Z]
        # (B, 768) + (B, 8) + (B, 16) -> (B, 792)
        fusion_input = torch.cat([cls_emb, one_hot_c, z], dim=-1)

        # 3. Predict Reward
        r = self.head(fusion_input) # (B, 1)

        return r.squeeze(-1) # (B,)
