"""Existing neural-network structures and inference-only mathematical helpers."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class CausalGRURegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.sequence_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )

        head_layers: list[nn.Module] = []
        last_dim = self.hidden_size + self.current_input_dim
        for hidden_dim in head_hidden_sizes:
            if hidden_dim <= 0:
                raise ValueError("head_hidden_sizes entries must be positive")
            head_layers.append(nn.Linear(last_dim, int(hidden_dim)))
            head_layers.append(nn.ReLU())
            if dropout > 0.0:
                head_layers.append(nn.Dropout(float(dropout)))
            last_dim = int(hidden_dim)
        head_layers.append(nn.Linear(last_dim, self.output_dim))
        self.head = nn.Sequential(*head_layers)

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        _, hidden = self.gru(sequence_inputs)
        representation = hidden[-1]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class AdaptiveSpectrumLayer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int = 128,
        dropout: float = 0.1,
        max_frequency_bins: int | None = None,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if max_frequency_bins is not None and max_frequency_bins <= 0:
            raise ValueError("max_frequency_bins must be positive when provided")
        self.input_dim = int(input_dim)
        self.hidden_size = int(hidden_size)
        self.max_frequency_bins = None if max_frequency_bins is None else int(max_frequency_bins)
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim * 3, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Linear(self.hidden_size, self.input_dim),
            nn.Sigmoid(),
        )

    def forward(self, sequence_inputs: torch.Tensor) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.input_dim:
            raise ValueError(f"Expected input feature dim {self.input_dim}, got {sequence_inputs.shape[2]}")

        history_size = sequence_inputs.shape[1]
        spectrum = torch.fft.rfft(sequence_inputs, dim=1)
        retained_bins = spectrum.shape[1]
        if self.max_frequency_bins is not None:
            retained_bins = min(retained_bins, self.max_frequency_bins)
        retained = spectrum[:, :retained_bins, :]

        magnitude = torch.abs(retained)
        phase = torch.angle(retained)
        gate_inputs = torch.cat([magnitude, torch.cos(phase), torch.sin(phase)], dim=2)
        weights = self.gate(gate_inputs)
        weighted = retained * weights.to(dtype=retained.dtype)

        if retained_bins < spectrum.shape[1]:
            padded = torch.zeros_like(spectrum)
            padded[:, :retained_bins, :] = weighted
            weighted = padded
        reconstructed = torch.fft.irfft(weighted, n=history_size, dim=1)
        return sequence_inputs + reconstructed


class CausalGRUASLRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        asl_hidden_size: int = 128,
        asl_dropout: float = 0.1,
        asl_max_frequency_bins: int | None = None,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        self.asl = AdaptiveSpectrumLayer(
            input_dim=sequence_input_dim,
            hidden_size=asl_hidden_size,
            dropout=asl_dropout,
            max_frequency_bins=asl_max_frequency_bins,
        )
        self.regressor = CausalGRURegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        return self.regressor(self.asl(sequence_inputs), current_inputs)


def _make_mlp_layers(
    input_dim: int,
    output_dim: int,
    hidden_sizes: tuple[int, ...],
    *,
    dropout: float = 0.0,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_sizes:
        if hidden_dim <= 0:
            raise ValueError("hidden_sizes entries must be positive")
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(activation())
        if dropout > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class CausalLSTMRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        lstm_dropout = float(dropout) if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.sequence_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.head = _make_mlp_layers(
            self.hidden_size + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        _, (hidden, _) = self.lstm(sequence_inputs)
        representation = hidden[-1]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class _CausalConvBlock(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if input_channels <= 0 or output_channels <= 0:
            raise ValueError("input_channels and output_channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        self.left_padding = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(
            int(input_channels),
            int(output_channels),
            kernel_size=int(kernel_size),
            dilation=int(dilation),
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity()
        self.residual = (
            nn.Identity()
            if int(input_channels) == int(output_channels)
            else nn.Conv1d(int(input_channels), int(output_channels), kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        padded = F.pad(inputs, (self.left_padding, 0))
        convolved = self.conv(padded)
        convolved = self.dropout(self.activation(convolved))
        return self.activation(convolved + self.residual(inputs))


class CausalTCNRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        channels: int = 128,
        num_blocks: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.channels = int(channels)
        self.num_blocks = int(num_blocks)
        self.kernel_size = int(kernel_size)
        blocks: list[nn.Module] = []
        input_channels = self.sequence_input_dim
        for block_idx in range(self.num_blocks):
            blocks.append(
                _CausalConvBlock(
                    input_channels=input_channels,
                    output_channels=self.channels,
                    kernel_size=self.kernel_size,
                    dilation=2**block_idx,
                    dropout=dropout,
                )
            )
            input_channels = self.channels
        self.network = nn.Sequential(*blocks)
        self.head = _make_mlp_layers(
            self.channels + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def encode_sequence(self, sequence_inputs: torch.Tensor) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        temporal = sequence_inputs.transpose(1, 2)
        encoded = self.network(temporal)
        return encoded.transpose(1, 2)

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encode_sequence(sequence_inputs)
        representation = encoded[:, -1, :]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class _PhaseFiLM(nn.Module):
    def __init__(self, *, conditioner_dim: int, feature_dim: int, hidden_size: int = 32, scale: float = 0.1):
        super().__init__()
        if conditioner_dim <= 0:
            raise ValueError("conditioner_dim must be positive")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if scale < 0.0:
            raise ValueError("scale must be non-negative")
        self.conditioner_dim = int(conditioner_dim)
        self.feature_dim = int(feature_dim)
        self.hidden_size = int(hidden_size)
        self.scale = float(scale)
        self.net = nn.Sequential(
            nn.Linear(self.conditioner_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2 * self.feature_dim),
        )
        final = self.net[-1]
        if isinstance(final, nn.Linear):
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def forward(self, features: torch.Tensor, conditioner: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.net(conditioner)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return features * (1.0 + self.scale * torch.tanh(gamma)) + self.scale * torch.tanh(beta)


class CausalTransformerRegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        d_model: int = 64,
        num_layers: int = 1,
        num_heads: int = 4,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
        max_history_size: int = 512,
        film_mode: str = "none",
        phase_conditioning_indices: tuple[int, ...] | None = None,
        film_hidden_size: int = 32,
        film_scale: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        if sequence_input_dim <= 0:
            raise ValueError("sequence_input_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if dim_feedforward <= 0:
            raise ValueError("dim_feedforward must be positive")
        if max_history_size <= 0:
            raise ValueError("max_history_size must be positive")
        resolved_film_mode = (film_mode or "none").lower()
        if resolved_film_mode not in {"none", "head", "input"}:
            raise ValueError(f"Unknown film_mode: {film_mode}")
        resolved_phase_conditioning_indices = tuple(int(v) for v in (phase_conditioning_indices or ()))
        if resolved_film_mode != "none" and not resolved_phase_conditioning_indices:
            raise ValueError("phase_conditioning_indices are required when film_mode is enabled")
        for index in resolved_phase_conditioning_indices:
            if index < 0 or index >= int(sequence_input_dim):
                raise ValueError(f"phase conditioning index {index} out of bounds")
        if film_hidden_size <= 0:
            raise ValueError("film_hidden_size must be positive")
        if film_scale < 0.0:
            raise ValueError("film_scale must be non-negative")

        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dim_feedforward = int(dim_feedforward)
        self.max_history_size = int(max_history_size)
        self.film_mode = resolved_film_mode
        self.phase_conditioning_indices = resolved_phase_conditioning_indices
        self.film_hidden_size = int(film_hidden_size)
        self.film_scale = float(film_scale)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.input_projection = nn.Linear(self.sequence_input_dim, self.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_history_size, self.d_model))
        self.input_film: _PhaseFiLM | None = None
        self.head_film: _PhaseFiLM | None = None
        if self.film_mode == "input":
            self.input_film = _PhaseFiLM(
                conditioner_dim=len(self.phase_conditioning_indices),
                feature_dim=self.d_model,
                hidden_size=self.film_hidden_size,
                scale=self.film_scale,
            )
        elif self.film_mode == "head":
            self.head_film = _PhaseFiLM(
                conditioner_dim=len(self.phase_conditioning_indices),
                feature_dim=self.d_model,
                hidden_size=self.film_hidden_size,
                scale=self.film_scale,
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=float(dropout),
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.head = _make_mlp_layers(
            self.d_model + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        if sequence_inputs.ndim != 3:
            raise ValueError("sequence_inputs must have shape [batch, history, features]")
        if sequence_inputs.shape[2] != self.sequence_input_dim:
            raise ValueError(
                f"Expected sequence feature dim {self.sequence_input_dim}, got {sequence_inputs.shape[2]}"
            )
        history_size = sequence_inputs.shape[1]
        if history_size > self.max_history_size:
            raise ValueError(f"history size {history_size} exceeds max_history_size {self.max_history_size}")
        embedded = self.input_projection(sequence_inputs)
        if self.input_film is not None:
            conditioner = sequence_inputs[:, :, list(self.phase_conditioning_indices)]
            embedded = self.input_film(embedded, conditioner)
        if self.use_positional_encoding:
            embedded = embedded + self.position_embedding[:, :history_size, :]
        mask = torch.triu(
            torch.ones(history_size, history_size, dtype=torch.bool, device=sequence_inputs.device),
            diagonal=1,
        )
        encoded = self.encoder(embedded, mask=mask)
        representation = encoded[:, -1, :]
        if self.head_film is not None:
            conditioner = sequence_inputs[:, -1, list(self.phase_conditioning_indices)]
            representation = self.head_film(representation, conditioner)
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class CausalTCNGRURegressor(nn.Module):
    def __init__(
        self,
        *,
        sequence_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        tcn_channels: int = 128,
        tcn_num_blocks: int = 3,
        tcn_kernel_size: int = 3,
        gru_hidden_size: int = 128,
        gru_num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if gru_hidden_size <= 0:
            raise ValueError("gru_hidden_size must be positive")
        if gru_num_layers <= 0:
            raise ValueError("gru_num_layers must be positive")
        self.tcn = CausalTCNRegressor(
            sequence_input_dim=sequence_input_dim,
            current_input_dim=0,
            output_dim=output_dim,
            channels=tcn_channels,
            num_blocks=tcn_num_blocks,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            head_hidden_sizes=head_hidden_sizes,
        )
        self.sequence_input_dim = int(sequence_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.tcn_channels = int(tcn_channels)
        self.tcn_num_blocks = int(tcn_num_blocks)
        self.tcn_kernel_size = int(tcn_kernel_size)
        self.gru_hidden_size = int(gru_hidden_size)
        self.gru_num_layers = int(gru_num_layers)
        gru_dropout = float(dropout) if gru_num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.tcn_channels,
            hidden_size=self.gru_hidden_size,
            num_layers=self.gru_num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = _make_mlp_layers(
            self.gru_hidden_size + self.current_input_dim,
            self.output_dim,
            tuple(int(v) for v in head_hidden_sizes),
            dropout=dropout,
        )

    def forward(self, sequence_inputs: torch.Tensor, current_inputs: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.tcn.encode_sequence(sequence_inputs)
        _, hidden = self.gru(encoded)
        representation = hidden[-1]
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 2 or current_inputs.shape[1] != self.current_input_dim:
                raise ValueError(f"Expected current_inputs shape [batch, {self.current_input_dim}]")
            representation = torch.cat([representation, current_inputs], dim=1)
        return self.head(representation)


class SubsectionGRUWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        head_hidden_sizes: tuple[int, ...] = (128,),
    ):
        super().__init__()
        if context_input_dim <= 0 or rollout_input_dim <= 0 or output_dim <= 0:
            raise ValueError("context_input_dim, rollout_input_dim, and output_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("hidden_size and num_layers must be positive")

        self.context_input_dim = int(context_input_dim)
        self.rollout_input_dim = int(rollout_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        gru_dropout = float(dropout) if num_layers > 1 else 0.0
        self.context_encoder = nn.GRU(
            input_size=self.context_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.rollout_decoder = nn.GRU(
            input_size=self.rollout_input_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.head = _make_mlp_layers(
            self.hidden_size + self.current_input_dim,
            self.output_dim,
            head_hidden_sizes,
            dropout=dropout,
        )

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_inputs.ndim != 3 or context_inputs.shape[2] != self.context_input_dim:
            raise ValueError(f"Expected context_inputs shape [batch, history, {self.context_input_dim}]")
        if rollout_inputs.ndim != 3 or rollout_inputs.shape[2] != self.rollout_input_dim:
            raise ValueError(f"Expected rollout_inputs shape [batch, rollout, {self.rollout_input_dim}]")
        _, hidden = self.context_encoder(context_inputs)
        decoded, _ = self.rollout_decoder(rollout_inputs, hidden)
        if self.current_input_dim > 0:
            if current_inputs is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            if current_inputs.ndim != 3 or current_inputs.shape[:2] != rollout_inputs.shape[:2]:
                raise ValueError("current_inputs must have shape [batch, rollout, current_features]")
            if current_inputs.shape[2] != self.current_input_dim:
                raise ValueError(f"Expected current feature dim {self.current_input_dim}")
            decoded = torch.cat([decoded, current_inputs], dim=2)
        batch_size, rollout_size, feature_dim = decoded.shape
        flat = decoded.reshape(batch_size * rollout_size, feature_dim)
        return self.head(flat).reshape(batch_size, rollout_size, self.output_dim)


class DiscreteSUBNETWrenchRegressor(nn.Module):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        latent_size: int = 16,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        if context_input_dim <= 0 or rollout_input_dim <= 0 or output_dim <= 0:
            raise ValueError("context_input_dim, rollout_input_dim, and output_dim must be positive")
        if current_input_dim < 0:
            raise ValueError("current_input_dim must be non-negative")
        if latent_size <= 0:
            raise ValueError("latent_size must be positive")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")

        self.context_input_dim = int(context_input_dim)
        self.rollout_input_dim = int(rollout_input_dim)
        self.current_input_dim = int(current_input_dim)
        self.output_dim = int(output_dim)
        self.latent_size = int(latent_size)
        encoder_hidden = int(hidden_sizes[0])
        self.context_encoder = nn.GRU(
            input_size=self.context_input_dim,
            hidden_size=encoder_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.encoder_to_latent = nn.Linear(encoder_hidden, self.latent_size)
        self.transition_net = _make_mlp_layers(
            self.latent_size + self.rollout_input_dim,
            self.latent_size,
            hidden_sizes,
            dropout=dropout,
        )
        self.output_net = _make_mlp_layers(
            self.latent_size + self.rollout_input_dim + self.current_input_dim,
            self.output_dim,
            hidden_sizes,
            dropout=dropout,
        )
        self.last_latent_rms = 0.0
        self.last_delta_latent_rms = 0.0

    def _initial_latent(self, context_inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.context_encoder(context_inputs)
        return self.encoder_to_latent(hidden[-1])

    def _step_delta(self, latent: torch.Tensor, rollout_input: torch.Tensor) -> torch.Tensor:
        return self.transition_net(torch.cat([latent, rollout_input], dim=1))

    def _output(
        self,
        latent: torch.Tensor,
        rollout_input: torch.Tensor,
        current_input: torch.Tensor | None,
    ) -> torch.Tensor:
        parts = [latent, rollout_input]
        if self.current_input_dim > 0:
            if current_input is None:
                raise ValueError("current_inputs are required when current_input_dim is positive")
            parts.append(current_input)
        return self.output_net(torch.cat(parts, dim=1))

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_inputs.ndim != 3 or context_inputs.shape[2] != self.context_input_dim:
            raise ValueError(f"Expected context_inputs shape [batch, history, {self.context_input_dim}]")
        if rollout_inputs.ndim != 3 or rollout_inputs.shape[2] != self.rollout_input_dim:
            raise ValueError(f"Expected rollout_inputs shape [batch, rollout, {self.rollout_input_dim}]")
        if self.current_input_dim > 0:
            if current_inputs is None or current_inputs.ndim != 3 or current_inputs.shape[:2] != rollout_inputs.shape[:2]:
                raise ValueError("current_inputs must have shape [batch, rollout, current_features]")
            if current_inputs.shape[2] != self.current_input_dim:
                raise ValueError(f"Expected current feature dim {self.current_input_dim}")

        latent = self._initial_latent(context_inputs)
        outputs: list[torch.Tensor] = []
        latent_values: list[torch.Tensor] = []
        delta_values: list[torch.Tensor] = []
        for step in range(rollout_inputs.shape[1]):
            rollout_step = rollout_inputs[:, step, :]
            current_step = current_inputs[:, step, :] if current_inputs is not None and self.current_input_dim > 0 else None
            outputs.append(self._output(latent, rollout_step, current_step))
            delta = self._step_delta(latent, rollout_step)
            latent_values.append(latent)
            delta_values.append(delta)
            latent = latent + delta
        with torch.no_grad():
            self.last_latent_rms = float(torch.sqrt(torch.mean(torch.square(torch.stack(latent_values)))) .detach().cpu())
            self.last_delta_latent_rms = float(torch.sqrt(torch.mean(torch.square(torch.stack(delta_values)))) .detach().cpu())
        return torch.stack(outputs, dim=1)


class ContinuousTimeSUBNETWrenchRegressor(DiscreteSUBNETWrenchRegressor):
    def __init__(
        self,
        *,
        context_input_dim: int,
        rollout_input_dim: int,
        current_input_dim: int,
        output_dim: int,
        latent_size: int = 16,
        hidden_sizes: tuple[int, ...] = (128, 128),
        dropout: float = 0.0,
        dt_over_tau: float = 0.03,
        integrator: str = "euler",
    ):
        if dt_over_tau <= 0.0 or not math.isfinite(float(dt_over_tau)):
            raise ValueError("dt_over_tau must be positive and finite")
        if integrator != "euler":
            raise ValueError("Only euler integrator is currently supported")
        super().__init__(
            context_input_dim=context_input_dim,
            rollout_input_dim=rollout_input_dim,
            current_input_dim=current_input_dim,
            output_dim=output_dim,
            latent_size=latent_size,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
        )
        self.dt_over_tau = float(dt_over_tau)
        self.integrator = integrator
        self.last_latent_derivative_rms = 0.0

    def forward(
        self,
        context_inputs: torch.Tensor,
        rollout_inputs: torch.Tensor,
        current_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_inputs.ndim != 3 or context_inputs.shape[2] != self.context_input_dim:
            raise ValueError(f"Expected context_inputs shape [batch, history, {self.context_input_dim}]")
        if rollout_inputs.ndim != 3 or rollout_inputs.shape[2] != self.rollout_input_dim:
            raise ValueError(f"Expected rollout_inputs shape [batch, rollout, {self.rollout_input_dim}]")
        if self.current_input_dim > 0:
            if current_inputs is None or current_inputs.ndim != 3 or current_inputs.shape[:2] != rollout_inputs.shape[:2]:
                raise ValueError("current_inputs must have shape [batch, rollout, current_features]")
            if current_inputs.shape[2] != self.current_input_dim:
                raise ValueError(f"Expected current feature dim {self.current_input_dim}")

        latent = self._initial_latent(context_inputs)
        outputs: list[torch.Tensor] = []
        latent_values: list[torch.Tensor] = []
        derivative_values: list[torch.Tensor] = []
        for step in range(rollout_inputs.shape[1]):
            rollout_step = rollout_inputs[:, step, :]
            current_step = current_inputs[:, step, :] if current_inputs is not None and self.current_input_dim > 0 else None
            outputs.append(self._output(latent, rollout_step, current_step))
            derivative = self._step_delta(latent, rollout_step)
            latent_values.append(latent)
            derivative_values.append(derivative)
            latent = latent + self.dt_over_tau * derivative
        with torch.no_grad():
            self.last_latent_rms = float(torch.sqrt(torch.mean(torch.square(torch.stack(latent_values)))) .detach().cpu())
            self.last_latent_derivative_rms = float(
                torch.sqrt(torch.mean(torch.square(torch.stack(derivative_values)))).detach().cpu()
            )
            self.last_delta_latent_rms = self.dt_over_tau * self.last_latent_derivative_rms
        return torch.stack(outputs, dim=1)


def cyclic_catmull_rom_weights(
    phase_radians: torch.Tensor,
    *,
    num_control_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_control_points < 4:
        raise ValueError("num_control_points must be at least 4 for Catmull-Rom interpolation")

    phase = torch.remainder(phase_radians, 2.0 * math.pi)
    position = phase * (float(num_control_points) / (2.0 * math.pi))
    base_index = torch.floor(position).to(torch.long)
    t = position - base_index.to(position.dtype)
    t2 = t * t
    t3 = t2 * t

    indices = torch.stack(
        [
            torch.remainder(base_index - 1, num_control_points),
            torch.remainder(base_index, num_control_points),
            torch.remainder(base_index + 1, num_control_points),
            torch.remainder(base_index + 2, num_control_points),
        ],
        dim=1,
    )
    weights = torch.stack(
        [
            -0.5 * t + t2 - 0.5 * t3,
            1.0 - 2.5 * t2 + 1.5 * t3,
            0.5 * t + 2.0 * t2 - 1.5 * t3,
            -0.5 * t2 + 0.5 * t3,
        ],
        dim=1,
    )
    return indices, weights


class PhaseFunctionedLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_control_points: int = 6):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if num_control_points < 4:
            raise ValueError("num_control_points must be at least 4")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.num_control_points = int(num_control_points)
        self.weight_control_points = nn.Parameter(torch.empty(num_control_points, output_dim, input_dim))
        self.bias_control_points = nn.Parameter(torch.empty(num_control_points, output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for idx in range(self.num_control_points):
            nn.init.xavier_uniform_(self.weight_control_points[idx])
        nn.init.zeros_(self.bias_control_points)

    def forward(self, inputs: torch.Tensor, phase_radians: torch.Tensor) -> torch.Tensor:
        indices, weights = cyclic_catmull_rom_weights(
            phase_radians.to(dtype=inputs.dtype),
            num_control_points=self.num_control_points,
        )
        selected_weights = self.weight_control_points[indices]
        selected_biases = self.bias_control_points[indices]
        interpolated_weights = torch.sum(weights[:, :, None, None] * selected_weights, dim=1)
        interpolated_biases = torch.sum(weights[:, :, None] * selected_biases, dim=1)
        return torch.einsum("boi,bi->bo", interpolated_weights, inputs) + interpolated_biases


class HybridPFNNRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: tuple[int, ...] = (40, 40),
        *,
        phase_feature_index: int = 0,
        expanded_input_dim: int = 45,
        phase_node_count: int = 5,
        phase_control_points: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(hidden_sizes) < 2:
            raise ValueError("HybridPFNNRegressor requires at least two hidden sizes")
        if input_dim <= 1:
            raise ValueError("input_dim must include phase plus at least one state feature")
        if phase_feature_index < 0 or phase_feature_index >= input_dim:
            raise ValueError("phase_feature_index is out of range")
        if expanded_input_dim <= 0 or phase_node_count < 0:
            raise ValueError("expanded_input_dim must be positive and phase_node_count must be non-negative")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.phase_feature_index = int(phase_feature_index)
        self.state_dim = int(input_dim - 1)
        self.expanded_input_dim = int(expanded_input_dim)
        self.phase_node_count = int(phase_node_count)
        self.phase_control_points = int(phase_control_points)

        first_hidden = int(hidden_sizes[0])
        second_hidden = int(hidden_sizes[1])
        self.input_expansion = nn.Sequential(nn.Linear(self.state_dim, self.expanded_input_dim), nn.ELU())
        self.phase_node_control_points = nn.Parameter(torch.empty(self.phase_control_points, self.phase_node_count))
        nn.init.normal_(self.phase_node_control_points, mean=0.0, std=0.02)

        first_input_dim = self.expanded_input_dim + self.phase_node_count + self.state_dim
        self.first_layer = nn.Linear(first_input_dim, first_hidden)
        self.phase_layer = PhaseFunctionedLinear(first_hidden + self.state_dim, second_hidden, self.phase_control_points)
        self.output_layer = nn.Linear(second_hidden + self.state_dim, output_dim)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _split_phase_and_state(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phase = inputs[:, self.phase_feature_index]
        if self.phase_feature_index == 0:
            state = inputs[:, 1:]
        elif self.phase_feature_index == inputs.shape[1] - 1:
            state = inputs[:, :-1]
        else:
            state = torch.cat([inputs[:, : self.phase_feature_index], inputs[:, self.phase_feature_index + 1 :]], dim=1)
        return phase, state

    def _phase_nodes(self, phase_radians: torch.Tensor) -> torch.Tensor:
        if self.phase_node_count == 0:
            return phase_radians.new_empty((len(phase_radians), 0))
        indices, weights = cyclic_catmull_rom_weights(
            phase_radians,
            num_control_points=self.phase_control_points,
        )
        selected = self.phase_node_control_points[indices]
        return torch.sum(weights[:, :, None] * selected, dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        phase, state = self._split_phase_and_state(inputs)
        expanded = self.input_expansion(state)
        phase_nodes = self._phase_nodes(phase.to(dtype=inputs.dtype))
        hidden1_input = torch.cat([expanded, phase_nodes, state], dim=1)
        hidden1 = self.dropout(self.activation(self.first_layer(hidden1_input)))
        hidden2_input = torch.cat([hidden1, state], dim=1)
        hidden2 = self.dropout(self.activation(self.phase_layer(hidden2_input, phase)))
        output_input = torch.cat([hidden2, state], dim=1)
        return self.output_layer(output_input)
