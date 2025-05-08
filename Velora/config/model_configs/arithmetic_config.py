"""
Configuración específica para los expertos aritméticos de VELORA.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .base_config import BaseConfig

@dataclass
class ArithmeticConfig(BaseConfig):
    """Configuración para el experto aritmético y sus componentes."""
    
    # Dimensiones específicas
    math_hidden_dim: int = 512
    math_layers: int = 4
    
    # Operaciones soportadas
    operations: Dict[int, str] = field(default_factory=lambda: {
        0: "suma",
        1: "resta",
        2: "multiplicación",
        3: "división"
    })
    
    # Configuración de operaciones
    max_operand_value: float = 1000.0
    min_operand_value: float = -1000.0
    decimal_precision: int = 5
    handle_division_by_zero: bool = True
    
    # Características especiales
    use_symbolic_verification: bool = True
    numeric_representation_bits: int = 32
    overflow_detection: bool = True
    
    # Parámetros de rendimiento
    numerical_stability_epsilon: float = 1e-7
    
    # Parámetros de entrenamiento específicos
    math_learning_rate: float = 3e-4
    math_weight_decay: float = 1e-5
    math_batch_size: int = 64
    
    def __post_init__(self):
        """Inicialización posterior que ajusta parámetros derivados."""
        # Sobrescribir algunos parámetros base con valores específicos
        self.activation_function = "relu"  # Mejor para operaciones numéricas
        self.precision = "fp32"  # Mayor precisión para cálculos matemáticos
        
        # Ajustar dimensiones si se especificaron diferentes a las base
        if self.math_hidden_dim != self.hidden_dim:
            self.hidden_dim = self.math_hidden_dim