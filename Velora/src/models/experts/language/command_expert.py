"""
Experto de comandos para VELORA.

Este módulo implementa el experto especializado en procesamiento
de consultas en forma de comando o instrucción.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ...components.attention import SelfAttentionBlock


class CommandExpert(nn.Module):
    """
    Experto especializado en procesamiento de comandos.
    
    Implementa capacidades específicas para entender, analizar
    y procesar consultas en forma imperativa o instructiva.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Capas transformer especializadas para comandos
        self.transformer_layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                ff_dim=self.hidden_dim * config.ff_expansion_factor,
                dropout_rate=config.dropout_rate,
                activation=config.activation_function
            ) for _ in range(3)  # 3 capas especializadas
        ])
        
        # Detector de verbo imperativo
        self.imperative_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Clasificador de tipo de comando
        self.command_type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 5)  # Tipos: acción, búsqueda, creación, cálculo, otros
        )
        
        # Extractor de acción (verbo principal)
        self.action_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Extractor de objeto (sobre qué se actúa)
        self.object_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Estimador de complejidad del comando
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # 0: simple, 1: complejo
        )
        
        # Generador de representación de respuesta a comando
        self.response_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Concatena repr. acción y objeto
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Procesa un comando y genera representación orientada a ejecución.
        
        Args:
            hidden_states: Representación de entrada [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara de atención [batch_size, seq_len]
            return_attention: Si es True, devuelve pesos de atención
            
        Returns:
            Representación procesada y opcionalmente pesos de atención
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Guardar representación de entrada para uso posterior
        original_repr = hidden_states
        
        # Procesar a través de capas transformer
        sequence_repr = hidden_states
        attentions = []
        
        for layer in self.transformer_layers:
            sequence_repr, attention = layer(sequence_repr, attention_mask, return_attention=True)
            if attention is not None:
                attentions.append(attention)
        
        # Detectar verbo imperativo (ej. "calcula", "busca", "encuentra")
        imperative_scores = self.imperative_detector(sequence_repr)  # [batch_size, seq_len, 1]
        
        # Aplicar máscara de atención si está disponible
        if attention_mask is not None:
            imperative_scores = imperative_scores * attention_mask.unsqueeze(-1)
        
        # Encontrar posición con mayor probabilidad de ser verbo imperativo
        verb_position = torch.argmax(imperative_scores.squeeze(-1), dim=-1)  # [batch_size]
        
        # Extraer representación de verbo (acción)
        # Usar representación del token con mayor score imperativo
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        action_repr = sequence_repr[batch_indices, verb_position]  # [batch_size, hidden_dim]
        
        # Mejorar representación de acción
        enhanced_action = self.action_extractor(action_repr)
        
        # Extraer objeto del comando (sobre qué se actúa)
        # Usar atención ponderada, excluyendo el verbo imperativo
        
        # Crear máscara para excluir el verbo
        verb_mask = torch.zeros_like(imperative_scores)
        verb_mask[batch_indices, verb_position] = 1.0
        object_weights = 1.0 - verb_mask
        
        # Normalizar pesos de objeto
        if attention_mask is not None:
            object_weights = object_weights * attention_mask.unsqueeze(-1)
        
        object_weights = object_weights / (object_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Aplicar atención ponderada
        object_repr = torch.bmm(
            object_weights.transpose(1, 2),  # [batch_size, 1, seq_len]
            sequence_repr  # [batch_size, seq_len, hidden_dim]
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        # Mejorar representación de objeto
        enhanced_object = self.object_extractor(object_repr)
        
        # Clasificar tipo de comando
        # Usar primer token [CLS] para clasificación
        cls_repr = sequence_repr[:, 0]
        command_type_logits = self.command_type_classifier(cls_repr)
        command_type_probs = F.softmax(command_type_logits, dim=-1)
        command_type = torch.argmax(command_type_probs, dim=-1)
        
        # Estimar complejidad
        complexity = self.complexity_estimator(cls_repr)
        
        # Generar representación para respuesta al comando
        # Concatenar representación de acción y objeto
        combined_repr = torch.cat([enhanced_action, enhanced_object], dim=1)
        response_repr = self.response_generator(combined_repr)
        
        # Expandir a formato de secuencia si es necesario para compatibilidad
        # con interfaz de otros expertos
        if original_repr.dim() > 2:
            response_repr = response_repr.unsqueeze(1)
        
        # Preparar salida de atención si se solicita
        final_attention = None
        if return_attention and attentions:
            # Devolver última capa de atención
            final_attention = attentions[-1]
        
        return response_repr, final_attention