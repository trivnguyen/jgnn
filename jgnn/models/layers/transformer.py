from typing import Optional

import torch
import torch.nn as nn


class MultiHeadAttentionBlock(nn.Module):
    """Multi-head attention block with pre-layer normalization.

    Uses pre-LN configuration (layer normalization within residual stream),
    which tends to provide better training stability than post-LN.

    Parameters
    ----------
    d_in : int
        Input dimension
    d_model : int
        Model embedding dimension
    d_mlp : int
        Hidden dimension of the feed-forward MLP
    n_heads : int
        Number of attention heads
    """
    def __init__(
        self,
        d_in: int,
        d_model: int,
        d_mlp: int,
        n_heads: int
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.n_heads = n_heads
        self.ln_attn1 = nn.LayerNorm(d_in)
        self.ln_attn2 = nn.LayerNorm(d_in)
        self.ln_mlp = nn.LayerNorm(d_in)
        self.attention = nn.MultiheadAttention(d_in, n_heads, batch_first=True)
        self.mlp1 = nn.Linear(d_in, d_mlp)
        self.mlp2 = nn.Linear(d_mlp, d_in)
        self.activation = nn.GELU()

        # Initialize attention weights
        nn.init.xavier_uniform_(self.attention.in_proj_weight)
        nn.init.zeros_(self.attention.in_proj_bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Multi-head attention
        if x is y:
            # Self-attention
            x_sa = self.ln_attn1(x)
            x_sa = self.attention(
                x_sa, x_sa, x_sa, key_padding_mask=mask, need_weights=False
            )[0]
        else:
            # Cross-attention
            x_sa = self.ln_attn1(x)
            y_sa = self.ln_attn2(y)
            x_sa = self.attention(
                x_sa, y_sa, y_sa, key_padding_mask=mask, need_weights=False
            )[0]

        # Add to residual stream
        x = x + x_sa

        # Feed-forward MLP with pre-LN
        x_mlp = self.ln_mlp(x)
        x_mlp = self.activation(self.mlp1(x_mlp))
        x_mlp = self.mlp2(x_mlp)

        # Add to residual stream
        x = x + x_mlp
        return x


class Transformer(nn.Module):
    """Decoder-only transformer for set modeling.

    Parameters
    ----------
    d_in : int
        Number of input (and output) features
    d_model : int
        Dimension of the model embedding space
    d_mlp : int
        Dimension of the feed-forward MLP
    n_layers : int
        Number of transformer layers
    n_heads : int
        Number of attention heads
    d_pos : int, optional
        Dimension of positional encoding features
    d_cond : int, optional
        Dimension of conditioning features
    concat_conditioning : bool
        Whether to concatenate conditioning to the input
    use_pos_enc : bool
        Whether to use positional encoding
    """
    def __init__(
        self,
        d_in: int,
        d_model: int = 128,
        d_mlp: int = 512,
        n_layers: int = 4,
        n_heads: int = 4,
        d_pos: Optional[int] = None,
        d_cond: Optional[int] = None,
        concat_conditioning: bool = False,
        use_pos_enc: bool = False
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.d_pos = d_pos
        self.d_cond = d_cond
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_pos_enc = use_pos_enc
        self.concat_conditioning = concat_conditioning

        self._setup_model()

    def _setup_model(self):
        # Input embedding layer
        self.input_embed = nn.Linear(self.d_in, self.d_model)

        # Positional encoding layer
        if self.use_pos_enc and self.d_pos is not None:
            self.pos_encoding_layer = nn.Linear(self.d_pos, self.d_model)
        else:
            self.pos_encoding_layer = None

        # Conditioning layers
        if self.d_cond is not None:
            self.conditioning_embed = nn.Linear(self.d_cond, self.d_model)
            self.conditioning_concat_embed = nn.Linear(
                self.d_model + self.d_in, self.d_model
            )
        else:
            self.conditioning_embed = None
            self.conditioning_concat_embed = None

        # Transformer layers
        self.transformer_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.transformer_layers.append(
                MultiHeadAttentionBlock(
                    self.d_model, self.d_model, self.d_mlp, self.n_heads
                )
            )

        # Final layer norm and output projection
        self.final_ln = nn.LayerNorm(self.d_model)
        self.unembed = nn.Linear(self.d_model, self.d_in)
        torch.nn.init.zeros_(self.unembed.weight)

    def forward(
        self,
        x: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pos_enc: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Input embedding
        x = self.input_embed(x)

        # Add positional encoding
        if pos_enc is not None and self.use_pos_enc:
            pos_enc = self.pos_encoding_layer(pos_enc)
            if mask is not None:
                pos_enc = torch.where(~mask[:, :, None], pos_enc, 0)
            x = x + pos_enc

        # Add conditioning
        if conditioning is not None:
            conditioning = self.conditioning_embed(conditioning)
            if self.concat_conditioning:
                conditioning = conditioning[:, None, :].repeat(1, x.size(1), 1)
                x = torch.cat([x, conditioning], dim=-1)
                x = self.conditioning_concat_embed(x)

        # Apply transformer layers
        for i in range(self.n_layers):
            if conditioning is not None and not self.concat_conditioning:
                x = x + conditioning[:, None, :]
            x = self.transformer_layers[i](x, x, mask, conditioning)

        # Final layer normalization
        x = self.final_ln(x)

        # Output projection
        x = self.unembed(x)

        return x
