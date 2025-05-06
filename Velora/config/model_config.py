# config/model_config.py
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuración para el modelo VELORA."""
    
    # Dimensiones
    input_dim: int = 128
    hidden_dim: int = 256
    latent_dim: int = 128
    
    # Configuraciones específicas de componentes
    router_hidden_dim: int = 256
    expert_hidden_dim: int = 256
    fusion_hidden_dim: int = 128
    
    # Parámetros generales
    num_experts: int = 2
    vocab_size: int = 10000
    
    # Memoria de trabajo
    use_memory: bool = False
    memory_size: int = 10
    
    # Opciones de entrenamiento
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    
    # Optimización
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1