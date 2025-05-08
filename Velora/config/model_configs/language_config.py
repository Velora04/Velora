"""
Configuración específica para los expertos lingüísticos de VELORA.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from .base_config import BaseConfig

@dataclass
class LanguageConfig(BaseConfig):
    """Configuración para el experto lingüístico y sus componentes."""
    
    # Dimensiones específicas
    language_hidden_dim: int = 768
    language_layers: int = 6
    max_seq_length: int = 512
    
    # Tipos de consultas soportadas
    query_types: Dict[int, str] = field(default_factory=lambda: {
        0: "pregunta",
        1: "comando",
        2: "declaración"
    })
    
    # Configuración del transformer
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    encoder_attention_heads: int = 12
    decoder_attention_heads: int = 12
    tie_word_embeddings: bool = True
    use_cache: bool = True
    
    # Generación de texto
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    max_new_tokens: int = 256
    min_new_tokens: int = 5
    repetition_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    
    # Parámetros de entrenamiento específicos
    language_learning_rate: float = 2e-4
    language_weight_decay: float = 1e-5
    language_batch_size: int = 32
    
    def __post_init__(self):
        """Inicialización posterior que ajusta parámetros derivados."""
        # Sobrescribir algunos parámetros base con valores específicos
        self.activation_function = "gelu"  # Mejor para procesamiento de lenguaje
        
        # Ajustar dimensiones si se especificaron diferentes a las base
        if self.language_hidden_dim != self.hidden_dim:
            self.hidden_dim = self.language_hidden_dim