"""
Módulo de fusión para VELORA.

Este módulo implementa el sistema de fusión que integra las salidas de 
diferentes expertos para producir una respuesta coherente.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any


class ExpertFusion(nn.Module):
    """
    Módulo de fusión que integra las salidas de diferentes expertos.
    
    Ofrece:
    1. Fusión adaptativa basada en confianza
    2. Resolución de conflictos
    3. Verificación de consistencia
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_experts = getattr(config, 'num_experts', 2)  # Por defecto: aritmético y lenguaje
        
        # Codificador para cada salida de experto
        self.expert_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # Mecanismo de atención para fusión inteligente
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=4,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Red de fusión para combinar representaciones
        self.fusion_network = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        )
        
        # Verificador de consistencia
        self.consistency_checker = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Inicialización
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos de las capas lineales."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        expert_outputs: List[torch.Tensor],
        expert_weights: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Fusiona salidas de expertos en una representación unificada.
        
        Args:
            expert_outputs: Lista de tensores de salida de expertos
            expert_weights: Pesos para cada experto [batch_size, num_experts]
            
        Returns:
            Representación fusionada y metadatos
        """
        device = expert_weights.device
        batch_size = expert_weights.size(0)
        
        # Normalizar pesos de expertos
        normalized_weights = F.normalize(expert_weights, p=1, dim=1)
        
        # Procesar cada salida de experto
        processed_outputs = []
        for output in expert_outputs:
            # Asegurar dimensión de batch
            if output.dim() == 1:
                output = output.unsqueeze(0)
                
            # Si es secuencia, reducirla a vector para fusión
            if output.dim() > 2:
                output = output.mean(dim=1)
                
            # Codificar salida
            processed = self.expert_encoder(output)
            processed_outputs.append(processed)
        
        # Si hay más de una salida, aplicar fusión con atención
        if len(processed_outputs) > 1:
            # Apilar salidas para atención
            stacked = torch.stack(processed_outputs, dim=1)  # [batch_size, num_experts, hidden_dim]
            
            # Crear consulta utilizando el primer experto como base
            # (en una implementación más sofisticada, podríamos usar un promedio ponderado)
            query = processed_outputs[0].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Aplicar atención para fusionar expertos
            attn_output, attn_weights = self.fusion_attention(query, stacked, stacked)
            
            # Preparar para red de fusión
            fusion_input = attn_output.squeeze(1)  # [batch_size, hidden_dim]
        else:
            # Si solo hay un experto, usar su salida directamente
            fusion_input = processed_outputs[0]
            attn_weights = torch.ones(batch_size, 1, 1, device=device)  # Atención ficticia
        
        # Aplicar red de fusión
        fused_output = self.fusion_network(fusion_input)
        
        # Verificar consistencia si hay múltiples expertos
        consistency_score = torch.ones((batch_size, 1), device=device) * 0.95  # Alta por defecto
        
        if len(expert_outputs) > 1:
            # Concatenar primeras dos salidas para verificación
            # (en sistemas más grandes, podríamos hacer comparaciones por pares)
            concat_outputs = torch.cat([processed_outputs[0], processed_outputs[1]], dim=1)
            
            # Redimensionar para consistencia si es necesario
            if concat_outputs.dim() > 2:
                concat_outputs = concat_outputs.reshape(batch_size, -1)
                
            # Calcular score de consistencia
            concat_dim = concat_outputs.size(1)
            if concat_dim == self.hidden_dim * 2:
                consistency_score = self.consistency_checker(concat_outputs)
            else:
                # Si las dimensiones no coinciden, manejar el caso (simplificado aquí)
                consistency_score = torch.ones((batch_size, 1), device=device) * 0.8
        
        # Compilar metadatos
        metadata = {
            'consistency_score': consistency_score,
            'expert_influence': normalized_weights,
            'attention_weights': attn_weights
        }
        
        return fused_output, metadata
    
    def resolve_conflicts(
        self,
        expert_outputs: List[torch.Tensor],
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """
        Resuelve conflictos entre expertos usando confianza.
        
        Args:
            expert_outputs: Lista de salidas en conflicto
            confidences: Confianza de cada salida [num_experts]
            
        Returns:
            Representación resuelta
        """
        # Encontrar el experto más confiable
        max_confidence, max_idx = torch.max(confidences, dim=0)
        most_confident_output = expert_outputs[max_idx]
        
        # Si la confianza máxima es baja, aplicar fusión ponderada
        if max_confidence < 0.7:
            # Normalizar confianzas
            weights = F.normalize(confidences, p=1, dim=0)
            
            # Fusionar ponderadamente
            weighted_sum = torch.zeros_like(expert_outputs[0])
            for i, output in enumerate(expert_outputs):
                weighted_sum += output * weights[i]
                
            return weighted_sum
        else:
            # Si hay confianza alta, usar el experto más confiable
            return most_confident_output