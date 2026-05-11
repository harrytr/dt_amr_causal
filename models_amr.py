#!/usr/bin/env python3
"""
models_amr.py
=============

AMR-only temporal model: Edge-aware GraphSAGE encoder + Transformer over time.

Input:
  graphs: list[T] of PyG Batch objects

Output:
  [B, n_outputs]

This model expects:
  - g.x exists with fixed feature width (from convert_to_pt.py)
  - g.edge_attr optionally exists (weight + scaled edge type)

Action conditioning:
  - When enabled, the model combines the temporal graph/state embedding with an
    intervention/action embedding using an explicit interaction head.
  - It can also consume a separate explicit state-summary feature vector
    prepared by the dataset layer.
  - This allows the predicted score to depend on graph-state, summary-state,
    and action interactions, rather than only on an additive state term plus
    an additive action term.
"""

from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Batch

try:
    from torch_geometric.nn import MessagePassing
except Exception:  # pragma: no cover
    from torch_geometric.nn.conv import MessagePassing


class EdgeSAGEConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: int = 0, aggr: str = "mean"):
        super().__init__(aggr=aggr)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.edge_dim = int(edge_dim)

        msg_in = self.in_channels + (self.edge_dim if self.edge_dim > 0 else 0)
        self.lin_neigh = nn.Linear(msg_in, self.out_channels, bias=True)
        self.lin_root = nn.Linear(self.in_channels, self.out_channels, bias=True)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = out + self.lin_root(x)
        return out

    def message(self, x_j, edge_attr):
        if self.edge_dim > 0 and edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            if edge_attr.size(-1) != self.edge_dim:
                edge_attr = edge_attr[:, : self.edge_dim]
            msg = torch.cat([x_j, edge_attr], dim=-1)
        else:
            msg = x_j
        return self.lin_neigh(msg)


class NodeAttentionPool(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        self.proj = nn.Linear(in_channels, hidden_channels)
        self.score = nn.Linear(hidden_channels, 1)

    def forward(self, x, batch, return_attention: bool = False):
        h = torch.tanh(self.proj(x))
        s = self.score(h).squeeze(-1)

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        out = torch.zeros_like(x)
        attn = torch.zeros((x.size(0),), device=x.device, dtype=x.dtype)

        for g in range(num_graphs):
            idx = (batch == g).nonzero(as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue
            w = F.softmax(s[idx], dim=0)
            attn[idx] = w
            out[idx] = x[idx] * w.unsqueeze(-1)

        pooled = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        pooled.index_add_(0, batch, out)
        if return_attention:
            return pooled, attn
        return pooled


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        dropout: float,
        n_layers: int = 2,
    ):
        super().__init__()
        self.dropout = float(dropout)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(EdgeSAGEConv(in_channels, hidden_channels, edge_dim=edge_dim))
        self.norms.append(nn.LayerNorm(hidden_channels))

        for _ in range(n_layers - 1):
            self.convs.append(EdgeSAGEConv(hidden_channels, hidden_channels, edge_dim=edge_dim))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.pool = NodeAttentionPool(hidden_channels)

    def forward(self, x, edge_index, edge_attr, batch, return_attention: bool = False):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pool(x, batch, return_attention=return_attention)


class AMRDyGFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        edge_dim: int,
        heads: int,
        T: int,
        dropout: float,
        use_cls_token: bool,
        n_outputs: int,
        n_layers: int = 2,
        sage_layers: int = 2,
        use_softplus: bool = True,
        output_activation: Optional[str] = None,
        action_feature_dim: int = 0,
        action_hidden_dim: int = 32,
        use_action_conditioning: bool = False,
        action_interaction_hidden_dim: int = 128,
        action_interaction_dropout: float = 0.1,
        state_summary_feature_dim: int = 0,
        state_summary_hidden_dim: int = 64,
    ):
        super().__init__()

        self.T = int(T)
        self.hidden_channels = int(hidden_channels)
        self.use_cls_token = bool(use_cls_token)
        self.use_softplus = bool(use_softplus)

        if output_activation is None:
            self.output_activation = "softplus" if self.use_softplus else "identity"
        else:
            self.output_activation = str(output_activation).lower().strip()

        self.use_action_conditioning = bool(use_action_conditioning)
        self.action_feature_dim = int(action_feature_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_interaction_hidden_dim = int(action_interaction_hidden_dim)
        self.action_interaction_dropout = float(action_interaction_dropout)
        self.state_summary_feature_dim = int(state_summary_feature_dim)
        self.state_summary_hidden_dim = int(state_summary_hidden_dim)

        self.gnn = GraphSAGEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            edge_dim=edge_dim,
            dropout=dropout,
            n_layers=sage_layers,
        )

        self.pos_emb = nn.Parameter(torch.randn(self.T + (1 if self.use_cls_token else 0), hidden_channels))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_channels))
        else:
            self.cls_token = None

        if self.use_action_conditioning:
            if self.action_feature_dim <= 0:
                raise ValueError("action_feature_dim must be > 0 when use_action_conditioning=True")

            self.action_proj = nn.Sequential(
                nn.LayerNorm(self.action_feature_dim),
                nn.Linear(self.action_feature_dim, self.action_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.action_hidden_dim),
            )

            if self.action_hidden_dim != self.hidden_channels:
                self.state_proj = nn.Sequential(
                    nn.Linear(self.hidden_channels, self.action_hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(self.action_hidden_dim),
                )
                interaction_dim = self.action_hidden_dim
            else:
                self.state_proj = nn.Identity()
                interaction_dim = self.hidden_channels

            if self.state_summary_feature_dim > 0:
                summary_hidden = self.state_summary_hidden_dim if self.state_summary_hidden_dim > 0 else interaction_dim
                self.state_summary_proj = nn.Sequential(
                    nn.LayerNorm(self.state_summary_feature_dim),
                    nn.Linear(self.state_summary_feature_dim, summary_hidden),
                    nn.ReLU(),
                    nn.LayerNorm(summary_hidden),
                    nn.Linear(summary_hidden, interaction_dim),
                    nn.ReLU(),
                    nn.LayerNorm(interaction_dim),
                )
                head_in = interaction_dim * 8
            else:
                self.state_summary_proj = None
                head_in = interaction_dim * 4

            self.head = nn.Sequential(
                nn.Linear(head_in, self.action_interaction_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(self.action_interaction_hidden_dim),
                nn.Dropout(self.action_interaction_dropout),
                nn.Linear(self.action_interaction_hidden_dim, n_outputs),
            )
        else:
            self.action_proj = None
            self.state_proj = None
            self.state_summary_proj = None
            self.head = nn.Linear(hidden_channels, n_outputs)

    def encode_batched_graph(self, g: Batch, return_attention: bool = False):
        x = g.x
        edge_index = g.edge_index
        edge_attr = getattr(g, "edge_attr", None)
        batch = g.batch
        return self.gnn(x, edge_index, edge_attr, batch, return_attention=return_attention)

    def encode_day_graph(self, g: Batch, return_attention: bool = False):
        x = g.x
        edge_index = g.edge_index
        edge_attr = getattr(g, "edge_attr", None)
        batch = g.batch
        return self.gnn(x, edge_index, edge_attr, batch, return_attention=return_attention)

    def encode_day_graphs(self, graphs: List[Batch], return_attention: bool = False):
        day_embs = []
        day_attn = []

        for g in graphs:
            if return_attention:
                pooled, attn = self.encode_day_graph(g, return_attention=True)
                day_embs.append(pooled)
                day_attn.append(attn)
            else:
                pooled = self.encode_day_graph(g, return_attention=False)
                day_embs.append(pooled)

        H = torch.stack(day_embs, dim=1)
        if return_attention:
            return H, day_attn
        return H

    def forward_from_day_embeddings(
        self,
        H: torch.Tensor,
        action_features: Optional[torch.Tensor] = None,
        state_summary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        H : torch.Tensor
            Shape [B, T, hidden_channels].
        action_features : Optional[torch.Tensor]
            Shape [B, F] when action conditioning is enabled.
        state_summary_features : Optional[torch.Tensor]
            Shape [B, S] when explicit state-summary features are available.

        Returns
        -------
        torch.Tensor
            Shape [B, n_outputs].
        """
        if self.use_cls_token:
            batch_size = H.size(0)
            cls = self.cls_token.expand(batch_size, -1, -1)
            H = torch.cat([cls, H], dim=1)

        H = H + self.pos_emb[: H.size(1)].unsqueeze(0)
        H = self.transformer(H)

        if self.use_cls_token:
            pooled = H[:, 0]
        else:
            pooled = H.mean(dim=1)

        if self.use_action_conditioning:
            if action_features is None:
                raise ValueError("action_features must be provided when use_action_conditioning=True")

            if action_features.dim() == 1:
                action_features = action_features.view(1, -1)

            if action_features.size(0) != pooled.size(0):
                raise ValueError(
                    f"action_features batch size mismatch: got {action_features.size(0)} "
                    f"expected {pooled.size(0)}"
                )

            action_features = action_features.to(device=pooled.device, dtype=pooled.dtype)

            graph_state_emb = self.state_proj(pooled)
            action_emb = self.action_proj(action_features)

            if self.state_summary_proj is not None:
                if state_summary_features is None:
                    raise ValueError(
                        "state_summary_features must be provided when state_summary_feature_dim > 0."
                    )

                if state_summary_features.dim() == 1:
                    state_summary_features = state_summary_features.view(1, -1)

                if state_summary_features.size(0) != pooled.size(0):
                    raise ValueError(
                        f"state_summary_features batch size mismatch: got {state_summary_features.size(0)} "
                        f"expected {pooled.size(0)}"
                    )

                state_summary_features = state_summary_features.to(device=pooled.device, dtype=pooled.dtype)
                summary_state_emb = self.state_summary_proj(state_summary_features)
                fused_state_emb = 0.5 * (graph_state_emb + summary_state_emb)

                graph_summary_prod = graph_state_emb * summary_state_emb
                graph_summary_diff = torch.abs(graph_state_emb - summary_state_emb)
                fused_action_prod = fused_state_emb * action_emb
                fused_action_diff = torch.abs(fused_state_emb - action_emb)

                joint = torch.cat(
                    [
                        graph_state_emb,
                        summary_state_emb,
                        fused_state_emb,
                        action_emb,
                        graph_summary_prod,
                        graph_summary_diff,
                        fused_action_prod,
                        fused_action_diff,
                    ],
                    dim=-1,
                )
            else:
                prod_emb = graph_state_emb * action_emb
                diff_emb = torch.abs(graph_state_emb - action_emb)
                joint = torch.cat([graph_state_emb, action_emb, prod_emb, diff_emb], dim=-1)

            out = self.head(joint)
        else:
            out = self.head(pooled)

        act = getattr(self, "output_activation", "softplus" if self.use_softplus else "identity")
        if act == "softplus":
            out = F.softplus(out)
        elif act == "sigmoid":
            out = torch.sigmoid(out)
        elif act == "identity" or act == "none":
            pass
        else:
            raise ValueError(f"Unknown output_activation: {act}")

        return out

    def forward(
        self,
        graphs: List[Batch],
        action_features: Optional[torch.Tensor] = None,
        state_summary_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H = self.encode_day_graphs(graphs, return_attention=False)
        return self.forward_from_day_embeddings(
            H,
            action_features=action_features,
            state_summary_features=state_summary_features,
        )