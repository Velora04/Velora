"""
Componentes de normalización especializados para VELORA.

Implementa capas de normalización adaptativas y personalizadas para
estabilizar el entrenamiento y mejorar el rendimiento del modelo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class AdaptiveLayerNorm(nn.Module):
    """
    Normalización de capa adaptativa que ajusta parámetros según el dominio.
    
    Permite que diferentes dominios (aritmético, lenguaje) tengan sus propios
    parámetros de normalización, facilitando la especialización.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_domains: int = 2,
        eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.eps = eps
        
        # Parámetros gamma (escala) para cada dominio
        self.gamma = nn.Parameter(torch.ones(num_domains, hidden_dim))
        
        # Parámetros beta (desplazamiento) para cada dominio
        self.beta = nn.Parameter(torch.zeros(num_domains, hidden_dim))
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización de parámetros."""
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
    
    def forward(
        self,
        x: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aplica normalización adaptativa según el dominio.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, hidden_dim]
            domain_ids: Tensor con identificadores de dominio [batch_size]
                        (si es None, se usa dominio 0 por defecto)
        
        Returns:
            Tensor normalizado
        """
        batch_size, seq_len, _ = x.size()
        
        # Calcular media y varianza por muestra
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        
        # Normalizar
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Si no se proporciona dominio, usar el primero por defecto
        if domain_ids is None:
            domain_ids = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        
        # Extraer parámetros específicos por dominio
        gamma_batch = self.gamma[domain_ids]  # [batch_size, hidden_dim]
        beta_batch = self.beta[domain_ids]    # [batch_size, hidden_dim]
        
        # Expandir para broadcasting con la secuencia
        gamma_batch = gamma_batch.unsqueeze(1).expand(-1, seq_len, -1)
        beta_batch = beta_batch.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Aplicar escalado y desplazamiento
        return gamma_batch * x_norm + beta_batch


class DomainNormalization(nn.Module):
    """
    Normalización entre dominios para alineación de representaciones.
    
    Facilita la transferencia de información entre expertos de diferentes
    dominios al normalizar representaciones a un espacio común.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_domains: int = 2,
        eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_domains = num_domains
        self.eps = eps
        
        # Proyecciones específicas de dominio a espacio común
        self.domain_projections = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_domains)
        ])
        
        # Normalización para espacio común
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización de parámetros."""
        for projection in self.domain_projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        source_domain: int,
        target_domain: Optional[int] = None
    ) -> torch.Tensor:
        """
        Normaliza una representación desde un dominio hacia otro.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, hidden_dim]
            source_domain: Dominio de origen (0 para aritmético, 1 para lenguaje)
            target_domain: Dominio destino (si es None, se normaliza a espacio común)
        
        Returns:
            Tensor normalizado al dominio objetivo
        """
        # Validar dominios
        if source_domain >= self.num_domains:
            raise ValueError(f"Dominio origen ({source_domain}) fuera de rango")
        
        if target_domain is not None and target_domain >= self.num_domains:
            raise ValueError(f"Dominio destino ({target_domain}) fuera de rango")
        
        # Proyectar desde dominio origen a espacio común
        common_space = self.domain_projections[source_domain](x)
        
        # Normalizar en espacio común
        common_space = self.norm(common_space)
        
        # Si no se especifica dominio destino, devolver representación en espacio común
        if target_domain is None:
            return common_space
        
        # Si es el mismo dominio, identidad con normalización
        if source_domain == target_domain:
            return common_space
        
        # Proyectar a dominio destino (inversa aproximada de la proyección origen)
        target_space = self.domain_projections[target_domain](common_space)
        
        return target_space


class GradientScaler(nn.Module):
    """
    Escala gradientes adaptativamente para estabilidad de entrenamiento.
    
    Permite controlar el flujo de gradientes entre componentes, facilitando
    el entrenamiento conjunto de partes con diferentes características.
    """
    
    def __init__(
        self,
        init_scale: float = 1.0,
        trainable: bool = True
    ):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale), requires_grad=trainable)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Aplica un factor de escala durante forward pass.
        El backward pass aplicará la inversa del escala a los gradientes.
        
        Args:
            x: Tensor de entrada
            
        Returns:
            Tensor escalado
        """
        return x * self.scale
    
    @property
    def current_scale(self) -> float:
        """Devuelve el valor actual del factor de escala."""
        return self.scale.item()


class ConditionalLayerNorm(nn.Module):
    """
    Normalización de capa condicional basada en contexto.
    
    Ajusta los parámetros de normalización según información de contexto,
    permitiendo adaptación dinámica a diferentes tipos de entradas.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        eps: float = 1e-5
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.eps = eps
        
        # Normalización base
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        
        # Redes para generar parámetros condicionales
        self.gamma_net = nn.Linear(context_dim, hidden_dim)
        self.beta_net = nn.Linear(context_dim, hidden_dim)
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicialización específica para parámetros condicionales."""
        # Iniciar gamma_net para producir valores cercanos a 1
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        
        # Iniciar beta_net para producir valores cercanos a 0
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor
    ) -> torch.Tensor:
        """
        Aplica normalización condicional basada en contexto.
        
        Args:
            x: Tensor de entrada [batch_size, seq_len, hidden_dim]
            context: Tensor de contexto [batch_size, context_dim]
        
        Returns:
            Tensor normalizado condicionalmente
        """
        # Normalización base
        # Normalización base
        x_norm = self.norm(x)
        
        # Generar parámetros condicionales
        gamma = self.gamma_net(context).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        beta = self.beta_net(context).unsqueeze(1)     # [batch_size, 1, hidden_dim]
        
        # Aplicar transformación condicional
        return gamma * x_norm + beta