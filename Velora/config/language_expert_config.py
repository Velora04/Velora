"""
Configuración específica para el experto de lenguaje de VELORA.
"""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LanguageExpertConfig:
    """Configuración para el experto de lenguaje."""
    
    # Dimensiones
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    
    # Arquitectura del modelo
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    activation: str = "gelu"  # 'relu', 'gelu', 'swish'
    
    # Vocabulario
    vocab_size: int = 10000
    max_sequence_length: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 2
    
    # Tipos de tareas soportadas
    task_types: List[str] = None  # Default: ['question', 'command', 'statement']
    
    # Capacidades lingüísticas
    support_classification: bool = True
    support_generation: bool = True
    support_understanding: bool = True
    
    # Opciones de entrenamiento
    learning_rate: float = 2e-4
    batch_size: int = 32
    num_epochs: int = 200
    warmup_steps: int = 1000
    
    # Regularización
    weight_decay: float = 1e-6
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    
    # Generación
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    max_new_tokens: int = 100
    
    # Mecanismos de atención
    use_causal_attention: bool = True
    use_rotary_embeddings: bool = False
    
    def __post_init__(self):
        """Inicializa valores por defecto después de la creación."""
        if self.task_types is None:
            self.task_types = ['question', 'command', 'statement']