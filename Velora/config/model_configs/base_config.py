"""
Configuración base para el modelo VELORA.
Define parámetros compartidos entre todos los componentes.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class BaseConfig:
    """Configuración base para todos los componentes de VELORA."""
    
    # Dimensiones fundamentales
    input_dim: int = 1024
    hidden_dim: int = 1024
    output_dim: int = 1024
    
    # Configuración del tokenizador
    vocab_size: int = 64000
    padding_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2
    domain_tokens: Dict[str, int] = field(default_factory=lambda: {
        "arithmetic": 3,
        "language": 4
    })
    
    # Parámetros de atención
    num_attention_heads: int = 16
    attention_dropout: float = 0.1
    use_rotary_embeddings: bool = True
    
    # Parámetros de feed-forward
    ff_expansion_factor: int = 4
    activation_function: str = "gelu"  # 'gelu', 'relu', 'swish'
    
    # Regularización
    dropout_rate: float = 0.1
    weight_decay: float = 1e-5
    gradient_clip_val: float = 1.0
    
    # Entrenamiento
    learning_rate: float = 2e-4
    warmup_steps: int = 1000
    batch_size: int = 32
    accumulation_steps: int = 1
    max_epochs: int = 100
    
    # Sistema de memoria
    use_memory: bool = True
    memory_size: int = 128
    memory_dim: int = 1024
    
    # Rutas y semilla
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Dispositivos
    device: str = "cuda"  # 'cuda', 'cpu', 'mps'
    precision: str = "fp16"  # 'fp32', 'fp16', 'bf16'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la configuración a diccionario."""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Crea una configuración desde un diccionario."""
        return cls(**config_dict)