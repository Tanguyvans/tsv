"""Prototype Network pour few-shot sur classes rares (FS-RSDD style).

Encodeur partagé (ResNet-50 timm) → embeddings L2-normalisés.
Prototype par classe = moyenne des embeddings du support set.
Prédiction = nearest prototype (cosine distance).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", embed_dim: int = 512, pretrained: bool = True):
        super().__init__()
        import timm
        self.encoder = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        feat_dim = self.encoder.num_features
        self.proj = nn.Linear(feat_dim, embed_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        f = self.proj(f)
        return F.normalize(f, dim=-1)

    def forward(
        self,
        support: torch.Tensor,
        support_labels: torch.Tensor,
        query: torch.Tensor,
        n_way: int,
    ) -> torch.Tensor:
        support_emb = self.embed(support)  # [N*K, D]
        query_emb = self.embed(query)      # [Q, D]
        prototypes = torch.stack([
            support_emb[support_labels == c].mean(0) for c in range(n_way)
        ])  # [N, D]
        prototypes = F.normalize(prototypes, dim=-1)
        # cosine similarity → logits
        return query_emb @ prototypes.T
