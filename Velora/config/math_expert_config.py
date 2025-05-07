"""
Configuración específica para el experto matemático de VELORA.
"""
from dataclasses import dataclass

@dataclass
class MathExpertConfig:
    """Configuración para el experto matemático."""
    
    # Dimensiones
    input_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 128
    
    # Arquitectura
    num_layers: int = 3
    activation: str = "relu"  # 'relu', 'gelu', 'swish'
    
    # Operaciones soportadas
    operations: list = None  # Default: [suma, resta, multiplicación, división]
    
    # Capacidades numéricas
    support_integers: bool = True
    support_decimals: bool = True
    support_fractions: bool = False
    support_scientific: bool = False
    
    # Rangos numéricos
    min_value: float = -1000.0
    max_value: float = 1000.0
    decimal_precision: int = 4
    
    # Opciones de entrenamiento
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 150
    
    # Regularización
    weight_decay: float = 1e-5
    dropout_rate: float = 0.1
    
    # Mecanismos de verificación
    use_symbolic_verification: bool = True
    verify_results: bool = True
    
    def __post_init__(self):
        """Inicializa valores por defecto después de la creación."""
        if self.operations is None:
            self.operations = [0, 1, 2, 3]  # 0=suma, 1=resta, 2=mult, 3=div