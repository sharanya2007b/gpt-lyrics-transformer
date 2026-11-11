
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:

    context_window: int = 1024
    vocabulary_size: int = 50257

    num_layers: int = 12
    num_heads: int = 12
    embedding_dim: int = 768

    dropout_rate: float = 0.1
    use_positional_embeddings: bool = True

    use_causal_mask: bool = True

    use_residual_connections: bool = True

    use_layernorm: bool = True

    ffn_multiplier: int = 4



class MaskedAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0

        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim
        self.head_dim = config.embedding_dim // config.num_heads
        self.training_config = config

        self.query_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

        self.key_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

        self.value_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

        self.output_projection = nn.Linear(
            config.embedding_dim,
            config.embedding_dim
        )

        self.attention_dropout = nn.Dropout(
            config.dropout_rate
        )

        self.output_dropout = nn.Dropout(
            config.dropout_rate
        )

        mask = torch.tril(
            torch.ones(
                config.context_window,
                config.context_window
            )
        )

        self.register_buffer(
            "causal_mask",
            mask.view(
                1,
                1,
                config.context_window,
                config.context_window
            )
        )

    def split_heads(self, tensor, batch, seq_len):

        return tensor.view(
            batch,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

    def forward(self, hidden_states):

        batch_size, seq_len, channels = hidden_states.size()

        Q = self.query_projection(hidden_states)
        K = self.key_projection(hidden_states)
        V = self.value_projection(hidden_states)

        Q = self.split_heads(Q, batch_size, seq_len)
        K = self.split_heads(K, batch_size, seq_len)
        V = self.split_heads(V, batch_size, seq_len)

        attention_scores = (
            Q @ K.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        if self.training_config.use_causal_mask:

            attention_scores = attention_scores.masked_fill(
                self.causal_mask[:, :, :seq_len, :seq_len] == 0,
                float("-inf")
            )



        attention_probs = F.softmax(
            attention_scores,
            dim=-1
        )

        attention_probs = self.attention_dropout(
            attention_probs
        )

        attention_output = attention_probs @ V

        attention_output = (
            attention_output
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, channels)
        )

        projected_output = self.output_projection(
            attention_output
        )

        projected_output = self.output_dropout(
            projected_output
        )

        self.latest_attention = attention_probs.detach()

        return projected_output


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()


        hidden_size = (
            config.ffn_multiplier
            * config.embedding_dim
        )



        self.network = nn.Sequential(
            nn.Linear(config.embedding_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, config.embedding_dim),
            nn.Dropout(config.dropout_rate)
        )

    def forward(self, x):

        return self.network(x)


class TransformerLayer(nn.Module):
    
    


    def __init__(self, config):
        super().__init__()

        self.attention_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.feedforward_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.attention = MaskedAttention(config)

        self.feedforward = FeedForward(config)

    def forward(self, hidden_states):

        if self.attention.training_config.use_layernorm:

            attention_input = self.attention_norm(
                hidden_states
            )

        else:

            attention_input = hidden_states

        attention_out = self.attention(
            attention_input
        )

        if self.attention.training_config.use_residual_connections:

            hidden_states = (
                hidden_states
                + attention_out
            )

        else:

            hidden_states = attention_out

        if self.attention.training_config.use_layernorm:

            ff_input = self.feedforward_norm(
                hidden_states
            )

        else:

            ff_input = hidden_states

        ff_out = self.feedforward(
            ff_input
        )

        if self.attention.training_config.use_residual_connections:

            hidden_states = (
                hidden_states
                + ff_out
            )

        else:

            hidden_states = ff_out

        return hidden_states


class NanoTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocabulary_size,
            config.embedding_dim
        )

        self.position_embedding = nn.Embedding(
            config.context_window,
            config.embedding_dim
        )

        self.embedding_dropout = nn.Dropout(
            config.dropout_rate
        )

        self.layers = nn.ModuleList([
            TransformerLayer(config)
            for _ in range(config.num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            config.embedding_dim
        )

        self.output_head = nn.Linear(
            config.embedding_dim,
            config.vocabulary_size,
            bias=False
        )

        self.output_head.weight = self.token_embedding.weight

        self.initialize_parameters()

        total_params = self.count_parameters()

        print(
            f"Total Parameters: "
            f"{total_params / 1e6:.2f}M"
        )

    def initialize_parameters(self):

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
                )

    def count_parameters(self):

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def forward(
        self,
        tokens,
        targets=None
    ):

        batch_size, seq_len = tokens.size()

        positions = torch.arange(
            0,
            seq_len,
            device=tokens.device
        )

        token_vectors = self.token_embedding(tokens)

        position_vectors = self.position_embedding(
            positions
        )

        if self.config.use_positional_embeddings:

            hidden_states = (
                token_vectors
                + position_vectors
            )

        else:

            hidden_states = token_vectors


        hidden_states = self.embedding_dropout(
            hidden_states
        )

        for layer in self.layers:

            hidden_states = layer(hidden_states)

        hidden_states = self.final_norm(
            hidden_states
        )

        token_logits = self.output_head(
            hidden_states
        )

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                token_logits.view(
                    -1,
                    token_logits.size(-1)
                ),
                targets.view(-1),
                ignore_index=-1
            )

        return token_logits, loss

    @torch.no_grad()
    def generate(
        self,
        tokens,
        max_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.0
    ):

        self.eval()

        for _ in range(max_tokens):

            context = (
                tokens[:, -self.config.context_window:]
            )

            logits, _ = self(context)

            next_token_logits = logits[:, -1, :]

            # Temperature scaling
            if temperature > 0:
                next_token_logits = (
                    next_token_logits / temperature
                )

            # Remove NaN / Inf
            next_token_logits = torch.nan_to_num(
                next_token_logits,
                nan=0.0,
                posinf=1e4,
                neginf=-1e4
            )

            # Repetition penalty
            if repetition_penalty != 1.0:

                for batch_index in range(tokens.shape[0]):

                    used_tokens = torch.unique(
                        tokens[batch_index]
                    )

                    for token_id in used_tokens:

                        token_id = int(token_id)

                        if (
                            next_token_logits[
                                batch_index,
                                token_id
                            ] > 0
                        ):

                            next_token_logits[
                                batch_index,
                                token_id
                            ] /= repetition_penalty

                        else:

                            next_token_logits[
                                batch_index,
                                token_id
                            ] *= repetition_penalty

            # Top-K filtering
            if top_k is not None:

                top_k = min(
                    top_k,
                    next_token_logits.size(-1)
                )

                values, _ = torch.topk(
                    next_token_logits,
                    top_k
                )

                threshold = values[:, [-1]]

                next_token_logits[
                    next_token_logits < threshold
                ] = -float("inf")

            # Top-P filtering
            if top_p is not None:

                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits,
                    descending=True
                )

                sorted_probs = F.softmax(
                    sorted_logits,
                    dim=-1
                )

                cumulative_probs = torch.cumsum(
                    sorted_probs,
                    dim=-1
                )

                remove_mask = cumulative_probs > top_p

                remove_mask[..., 1:] = remove_mask[
                    ..., :-1
                ].clone()

                remove_mask[..., 0] = 0

                sorted_logits[
                    remove_mask
                ] = -float("inf")

                next_token_logits = torch.full_like(
                    next_token_logits,
                    -float("inf")
                )

                next_token_logits.scatter_(
                    1,
                    sorted_indices,
                    sorted_logits
                )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1
            )

            # Remove NaN probabilities
            probabilities = torch.nan_to_num(
                probabilities,
                nan=0.0
            )

            # Normalize safely
            probability_sums = probabilities.sum(
                dim=-1,
                keepdim=True
            )

            probability_sums = torch.clamp(
                probability_sums,
                min=1e-8
            )

            probabilities = (
                probabilities / probability_sums
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            tokens = torch.cat(
                [tokens, next_token],
                dim=1
            )

        return tokens