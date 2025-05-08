"""
Mecanismos de atención personalizados para VELORA.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MultiheadAttention(nn.Module):
    """
    Implementación de atención multi-cabeza con adaptaciones para VELORA.
    
    Soporta atención estándar, enmascarada (causal), y extensiones para
    mayor eficiencia y calidad de atención.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1,
        is_causal: bool = False
    ):
        super().__init__()
        
        # Validar que la dimensión es divisible por número de cabezas
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.is_causal = is_causal
        
        # Proyecciones lineales
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout para regularización
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        
        # Inicialización de pesos
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización especial para mayor estabilidad."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calcula atención multi-cabeza.
        
        Args:
            query: Tensor de consulta [batch_size, seq_len_q, hidden_dim]
            key: Tensor de clave [batch_size, seq_len_k, hidden_dim]
            value: Tensor de valor [batch_size, seq_len_k, hidden_dim]
            attention_mask: Máscara opcional [batch_size, 1, seq_len_q, seq_len_k]
            return_attention: Si es True, devuelve pesos de atención
        
        Returns:
            Tensor de salida [batch_size, seq_len_q, hidden_dim] y
            opcionalmente pesos de atención [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size, q_len, _ = query.size()
        _, k_len, _ = key.size()
        
        # Proyectar query, key, value
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Dividir en múltiples cabezas y transposición
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcular scores de atención: (B, H, Lq, D) x (B, H, D, Lk) -> (B, H, Lq, Lk)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Aplicar máscara de atención si se proporciona
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Aplicar máscara causal si es necesario
        if self.is_causal:
            causal_mask = torch.triu(
                torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device),
                diagonal=1
            )
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), -10000.0)
        
        # Aplicar softmax y dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Obtener contexto: (B, H, Lq, Lk) x (B, H, Lk, D) -> (B, H, Lq, D)
        context = torch.matmul(attention_weights, v)
        
        # Transponer y reshapear: (B, H, Lq, D) -> (B, Lq, H*D)
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_dim)
        
        # Proyección final y dropout
        output = self.out_proj(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class CrossAttention(nn.Module):
    """
    Atención cruzada para relacionar representaciones de diferentes dominios.
    
    Permite la interacción entre expertos o la integración de diferentes
    fuentes de información.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Validación de dimensiones
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible por num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Proyecciones lineales
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Parámetros de escala para control de flujo
        self.scale_factor = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))
        
        # Dropout para regularización
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_dropout = nn.Dropout(dropout_rate)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización especial para mayor estabilidad."""
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query_domain: torch.Tensor,
        key_value_domain: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Calcula atención cruzada entre dominios diferentes.
        
        Args:
            query_domain: Tensor del dominio de consulta [batch_size, q_len, hidden_dim]
            key_value_domain: Tensor del dominio KV [batch_size, kv_len, hidden_dim]
            attention_mask: Máscara opcional [batch_size, 1, q_len, kv_len]
            return_attention: Si es True, devuelve pesos de atención
        
        Returns:
            Tensor de salida [batch_size, q_len, hidden_dim] y
            opcionalmente pesos de atención [batch_size, num_heads, q_len, kv_len]
        """
        batch_size, q_len, _ = query_domain.size()
        _, kv_len, _ = key_value_domain.size()
        
        # Proyectar query, key, value
        q = self.q_proj(query_domain)
        k = self.k_proj(key_value_domain)
        v = self.v_proj(key_value_domain)
        
        # Dividir en múltiples cabezas
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calcular scores de atención con factor de escala adaptable
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale_factor / math.sqrt(self.head_dim)
        
        # Aplicar máscara si se proporciona
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Aplicar softmax y dropout con parámetro beta para control
        attention_weights = F.softmax(scores + self.beta, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Obtener contexto
        context = torch.matmul(attention_weights, v)
        
        # Transponer y reshapear
        context = context.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_dim)
        
        # Proyección final
        output = self.out_proj(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attention_weights
        else:
            return output, None


class SelfAttentionBlock(nn.Module):
    """
    Bloque completo de auto-atención con feed-forward, normalización y residuales.
    
    Componente básico para múltiples expertos y enrutadores.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ff_dim: int = None,
        dropout_rate: float = 0.1,
        activation: str = "gelu",
        is_causal: bool = False,
        pre_norm: bool = True
    ):
        super().__init__()
        
        # Establecer dimensiones
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim if ff_dim is not None else 4 * hidden_dim
        self.pre_norm = pre_norm
        
        # Capas de atención
        self.self_attention = MultiheadAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_causal=is_causal
        )
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, self.ff_dim),
            self._get_activation_fn(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(self.ff_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        
        # Normalizaciones
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización para redes feed-forward."""
        for m in self.ff.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _get_activation_fn(self, activation: str):
        """Devuelve la función de activación correspondiente."""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Activación no soportada: {activation}")
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Procesa la entrada a través del bloque de atención completo.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara opcional [batch_size, 1, seq_len, seq_len]
            return_attention: Si es True, devuelve pesos de atención
            
        Returns:
            Tensor procesado y opcionalmente pesos de atención
        """
        attn_weights = None
        
        if self.pre_norm:
            # Pre-normalización
            residual = x
            x = self.norm1(x)
            x_attn, attn_weights = self.self_attention(x, x, x, attention_mask, return_attention)
            x = residual + x_attn
            
            residual = x
            x = self.norm2(x)
            x_ff = self.ff(x)
            x = residual + x_ff
        else:
            # Post-normalización
            residual = x
            x_attn, attn_weights = self.self_attention(x, x, x, attention_mask, return_attention)
            x = self.norm1(residual + x_attn)
            
            residual = x
            x_ff = self.ff(x)
            x = self.norm2(residual + x_ff)
        
        if return_attention:
            return x, attn_weights
        else:
            return x, None