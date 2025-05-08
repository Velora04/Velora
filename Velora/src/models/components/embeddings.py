"""
Implementación de capas de embedding para VELORA.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class VeloraEmbeddings(nn.Module):
    """
    Sistema unificado de embeddings para VELORA.
    
    Combina embeddings de token, posición y tipo (dominio) en una
    representación coherente utilizada por todos los componentes.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Dimensiones principales
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.dropout_rate = config.dropout_rate
        self.max_position_embeddings = getattr(config, "max_seq_length", 512)
        
        # Embeddings de token
        self.token_embeddings = nn.Embedding(
            self.vocab_size, 
            self.hidden_dim, 
            padding_idx=config.padding_token_id
        )
        
        # Embeddings posicionales (absolutos)
        self.position_embeddings = nn.Embedding(
            self.max_position_embeddings, 
            self.hidden_dim
        )
        
        # Embeddings de tipo para diferenciar dominios
        self.type_embeddings = nn.Embedding(
            len(config.domain_tokens) + 1,  # +1 para 'genérico'
            self.hidden_dim
        )
        
        # Normalización y dropout
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Registro de buffer para posiciones
        position_ids = torch.arange(self.max_position_embeddings).unsqueeze(0)
        self.register_buffer("position_ids", position_ids)
        
        # Inicialización especial para tokens numéricos
        self._initialize_numerical_embeddings()
    
    def _initialize_numerical_embeddings(self):
        """
        Inicialización especial para tokens numéricos para preservar
        propiedades numéricas en el espacio de embeddings.
        """
        # Asumimos que los tokens numéricos (0-9) están en posiciones específicas
        # Esto se debe ajustar según la implementación real del tokenizador
        numerical_token_indices = list(range(10, 20))  # Ejemplo: tokens 10-19 para dígitos 0-9
        
        # Inicialización que preserva distancia numérica
        for i, idx in enumerate(numerical_token_indices):
            if idx < self.vocab_size:
                # Vector base para todos los dígitos
                base_vector = torch.randn(self.hidden_dim) * 0.02
                # Componente numérica proporcional al valor
                numeric_component = torch.ones(self.hidden_dim) * (i / 10)
                # Combinar y normalizar
                combined = base_vector + numeric_component
                normalized = combined / combined.norm()
                # Asignar al embedding
                with torch.no_grad():
                    self.token_embeddings.weight[idx] = normalized * math.sqrt(self.hidden_dim)
    
    def forward(
        self, 
        input_ids: torch.Tensor, 
        domain_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Procesa los ids de entrada y genera embeddings completos.
        
        Args:
            input_ids: Tensor de índices de tokens [batch_size, seq_len]
            domain_type_ids: Tensor que indica el dominio (aritmético, lenguaje, etc.)
            position_ids: Tensor de posiciones (opcional)
            attention_mask: Máscara de atención para posiciones de padding
            
        Returns:
            Tensor de embeddings y máscara de atención
        """
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device
        
        # Generar información posicional si no se proporciona
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        
        # Generar tipos de dominio si no se proporcionan
        if domain_type_ids is None:
            # Por defecto, asumir dominio genérico (0)
            domain_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        
        # Generar máscara de atención si no se proporciona
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        
        # Obtener los distintos embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        domain_embeds = self.type_embeddings(domain_type_ids)
        
        # Combinar todos los embeddings
        embeddings = token_embeds + position_embeds + domain_embeds
        
        # Normalizar y aplicar dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Convertir máscara de atención a formato adecuado para transformers 
        # (1.0 para tokens a atender, 0.0 para ignorar)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return embeddings, extended_attention_mask


class RotaryPositionEmbedding(nn.Module):
    """
    Implementación de Embeddings Posicionales Rotatorios (RoPE).
    
    Proporciona codificación posicional dentro del espacio de atención
    mediante rotación de pares de dimensiones en el espacio de claves y consultas.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        # Crear frecuencias base
        freqs = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("freqs", freqs)
        
        # Crear buffer para posiciones
        t = torch.arange(self.max_seq_len, dtype=torch.float)
        self.register_buffer("t", t)
        
        # Calcular senos y cosenos para todas las posiciones
        freqs = torch.outer(t, freqs)
        self.register_buffer("sin", freqs.sin())  # [seq_len, dim/2]
        self.register_buffer("cos", freqs.cos())  # [seq_len, dim/2]
    
    def forward(self, x: torch.Tensor, seq_dim: int = 1) -> torch.Tensor:
        """
        Aplica embedding posicional rotacional.
        
        Args:
            x: Tensor a embedder de forma [batch_size, seq_len, dim]
            seq_dim: Dimensión que representa la secuencia (normalmente 1)
            
        Returns:
            Tensor con posiciones codificadas mediante rotación
        """
        seq_len = x.size(seq_dim)
        
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length ({seq_len}) exceeds maximum length ({self.max_seq_len})")
        
        x_shape = x.shape
        
        # Obtener dimensión de embedding
        dim = x.size(-1)
        
        # Asegurar que la dimensión es par para RoPE
        assert dim % 2 == 0, f"Dimension {dim} must be even for RoPE"
        
        # Reordenar para trabajar con pares
        x_rope = x.view(*x_shape[:-1], -1, 2)
        
        # Obtener senos y cosenos para la longitud de secuencia actual
        sin = self.sin[:seq_len]
        cos = self.cos[:seq_len]
        
        # Expandir senos y cosenos para compatibilidad con forma de x
        expand_shape = [1] * (len(x_shape) - 2) + [seq_len, dim // 2]
        sin = sin.view(*expand_shape)
        cos = cos.view(*expand_shape)
        
        # Mover dimensión seq_len al lugar correcto si no está en posición 1
        if seq_dim != 1:
            sin = sin.transpose(1, seq_dim)
            cos = cos.transpose(1, seq_dim)
        
        # Rotación en pares de dimensiones
        x1 = x_rope[..., 0]
        x2 = x_rope[..., 1]
        
        # Aplicar rotación:
        # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        result_x1 = x1 * cos - x2 * sin
        result_x2 = x1 * sin + x2 * cos
        
        # Reordenar y devolver
        result = torch.stack([result_x1, result_x2], dim=-1)
        return result.view(*x_shape)