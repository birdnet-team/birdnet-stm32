"""Model configuration dataclass with JSON serialization and validation.

Replaces the raw dict config previously saved alongside checkpoints.
Backward-compatible: loads legacy JSON files that lack newer fields.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Immutable model configuration for training, conversion, and deployment.

    Attributes:
        sample_rate: Audio sample rate in Hz.
        num_mels: Number of mel frequency bins.
        spec_width: Spectrogram width in frames.
        fft_length: FFT window size.
        chunk_duration: Audio chunk duration in seconds.
        hop_length: STFT hop length in samples.
        audio_frontend: Frontend mode ('librosa', 'hybrid', 'raw').
        mag_scale: Magnitude scaling ('pwl', 'none').
        input_compression: Compression of a precomputed spectrogram input
            before quantization ('none', 'sqrt', 'log'); librosa/hybrid only.
        embeddings_size: Dense embedding dimension before classifier.
        alpha: Width multiplier for channel counts.
        depth_multiplier: Block repeat count per stage.
        head_pooling: Pooling head ('gap', 'freq_mean_time_maxmean').
        dw_kernel_size: Depthwise kernel size in stages 2-4 (3 or 5).
        stage_widths: Base output channels of the four stages, before alpha.
        num_classes: Number of output classes.
        class_names: Ordered list of class label strings.
        frontend_trainable: Whether frontend weights are trainable.
        dropout_rate: Dropout rate before classifier head.
    """

    # Audio
    sample_rate: int = 24000
    num_mels: int = 64
    spec_width: int = 256
    fft_length: int = 512
    chunk_duration: float = 3.0
    hop_length: int = 281
    audio_frontend: str = "hybrid"
    mag_scale: str = "pwl"
    input_compression: str = "none"

    # Model architecture
    embeddings_size: int = 256
    alpha: float = 1.0
    depth_multiplier: int = 1
    head_pooling: str = "gap"
    dw_kernel_size: int = 3
    stage_widths: list[int] = field(default_factory=lambda: [32, 64, 128, 256])
    dropout_rate: float = 0.5
    frontend_trainable: bool = False

    # Classes
    num_classes: int = 0
    class_names: list[str] = field(default_factory=list)

    # -- Validation ----------------------------------------------------------

    _VALID_FRONTENDS = frozenset({"librosa", "hybrid", "raw"})
    _VALID_MAG_SCALES = frozenset({"pwl", "none"})
    _VALID_INPUT_COMPRESSIONS = frozenset({"none", "sqrt", "log"})
    _VALID_HEAD_POOLINGS = frozenset({"gap", "freq_mean_time_maxmean"})
    _VALID_DW_KERNEL_SIZES = frozenset({3, 5})

    def __post_init__(self) -> None:
        """Validate field values after initialization."""
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.num_mels <= 0:
            raise ValueError(f"num_mels must be positive, got {self.num_mels}")
        if self.spec_width <= 0:
            raise ValueError(f"spec_width must be positive, got {self.spec_width}")
        if self.fft_length <= 0:
            raise ValueError(f"fft_length must be positive, got {self.fft_length}")
        if self.chunk_duration <= 0:
            raise ValueError(f"chunk_duration must be positive, got {self.chunk_duration}")
        if self.audio_frontend not in self._VALID_FRONTENDS:
            raise ValueError(f"audio_frontend '{self.audio_frontend}' not in {sorted(self._VALID_FRONTENDS)}")
        if self.mag_scale not in self._VALID_MAG_SCALES:
            raise ValueError(f"mag_scale '{self.mag_scale}' not in {sorted(self._VALID_MAG_SCALES)}")
        if self.input_compression not in self._VALID_INPUT_COMPRESSIONS:
            raise ValueError(
                f"input_compression '{self.input_compression}' not in {sorted(self._VALID_INPUT_COMPRESSIONS)}"
            )
        if self.input_compression != "none" and self.audio_frontend == "raw":
            raise ValueError("input_compression applies to spectrogram inputs, not the raw frontend")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if self.depth_multiplier < 1:
            raise ValueError(f"depth_multiplier must be >= 1, got {self.depth_multiplier}")
        if self.head_pooling not in self._VALID_HEAD_POOLINGS:
            raise ValueError(f"head_pooling '{self.head_pooling}' not in {sorted(self._VALID_HEAD_POOLINGS)}")
        if self.dw_kernel_size not in self._VALID_DW_KERNEL_SIZES:
            raise ValueError(f"dw_kernel_size {self.dw_kernel_size} not in {sorted(self._VALID_DW_KERNEL_SIZES)}")
        if len(self.stage_widths) != 4 or any(int(w) <= 0 for w in self.stage_widths):
            raise ValueError(f"stage_widths must be four positive widths, got {self.stage_widths}")
        if not 0 <= self.dropout_rate < 1:
            raise ValueError(f"dropout_rate must be in [0, 1), got {self.dropout_rate}")
        if self.num_classes < 0:
            raise ValueError(f"num_classes must be >= 0, got {self.num_classes}")
        if self.class_names and len(self.class_names) != self.num_classes:
            raise ValueError(f"class_names length ({len(self.class_names)}) != num_classes ({self.num_classes})")

    # -- Serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for JSON serialization."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write config to a JSON file.

        Args:
            path: Destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        """Create a ModelConfig from a dict, ignoring unknown keys.

        Unknown keys are silently dropped so legacy configs still load.

        Args:
            data: Dictionary of config values.

        Returns:
            Validated ModelConfig instance.
        """
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """Load config from a JSON file.

        Args:
            path: Path to JSON config file.

        Returns:
            Validated ModelConfig instance.

        Raises:
            FileNotFoundError: If path does not exist.
        """
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)
