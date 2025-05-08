"""
Experto de declaraciones para VELORA.

Este módulo implementa el experto especializado en procesamiento
de consultas en forma de declaración o afirmación.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ...components.attention import SelfAttentionBlock


class StatementExpert(nn.Module):
    """
    Experto especializado en procesamiento de declaraciones.
    
    Implementa capacidades específicas para entender, analizar
    y procesar consultas en forma declarativa o informativa.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Capas transformer especializadas para declaraciones
        self.transformer_layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                ff_dim=self.hidden_dim * config.ff_expansion_factor,
                dropout_rate=config.dropout_rate,
                activation=config.activation_function
            ) for _ in range(3)  # 3 capas especializadas
        ])
        
        # Clasificador de tipo de declaración
        self.statement_type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 4)  # Tipos: factual, opinión, condicional, comparativa
        )
        
        # Extractor de sujeto de la declaración
        self.subject_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Extractor de predicado de la declaración
        self.predicate_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Detector de valoración (positiva/negativa/neutra)
        self.sentiment_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 3),  # 3 clases: negativo, neutro, positivo
            nn.Softmax(dim=-1)
        )
        
        # Estimador de certeza/confianza en la declaración
        self.certainty_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # 0: incierto, 1: certeza alta
        )
        
        # Generador de representación de respuesta a declaración
        self.response_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Concatena repr. sujeto y predicado
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
        Procesa una declaración y genera representación de análisis.
        
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
        
        # Clasificar tipo de declaración
        # Usar primer token [CLS] para clasificación
        cls_repr = sequence_repr[:, 0]
        statement_type_logits = self.statement_type_classifier(cls_repr)
        statement_type_probs = F.softmax(statement_type_logits, dim=-1)
        statement_type = torch.argmax(statement_type_probs, dim=-1)
        
        # Detectar sujeto y predicado utilizando atención y posición
        # En una implementación completa, esto se haría con un modelo de
        # parsing gramatical más sofisticado. Simplificamos con una aproximación:
        
        # Asumimos que el sujeto tiende a estar al inicio (primeros 30%)
        # y el predicado en el resto de la oración
        subject_end = max(1, int(seq_len * 0.3))
        
        # Extraer representación de sujeto (promedio de tokens iniciales)
        if attention_mask is not None:
            # Crear máscara para primeros tokens
            subject_mask = attention_mask.clone()
            subject_mask[:, subject_end:] = 0
            
            mask_expanded = subject_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            subject_sum = torch.sum(sequence_repr * mask_expanded, dim=1)
            subject_len = torch.sum(subject_mask, dim=1, keepdim=True).clamp(min=1)
            subject_repr = subject_sum / subject_len
        else:
            subject_repr = sequence_repr[:, :subject_end].mean(dim=1)
        
        # Extraer representación de predicado (resto de la oración)
        if attention_mask is not None:
            # Crear máscara para tokens posteriores
            predicate_mask = attention_mask.clone()
            predicate_mask[:, :subject_end] = 0
            
            mask_expanded = predicate_mask.unsqueeze(-1).expand(-1, -1, self.hidden_dim)
            predicate_sum = torch.sum(sequence_repr * mask_expanded, dim=1)
            predicate_len = torch.sum(predicate_mask, dim=1, keepdim=True).clamp(min=1)
            predicate_repr = predicate_sum / predicate_len
        else:
            predicate_repr = sequence_repr[:, subject_end:].mean(dim=1)
        
        # Mejorar representaciones
        enhanced_subject = self.subject_extractor(subject_repr)
        enhanced_predicate = self.predicate_extractor(predicate_repr)
        
        # Detectar valoración/sentimiento
        sentiment = self.sentiment_detector(cls_repr)
        
        # Estimar certeza/confianza en la declaración
        certainty = self.certainty_estimator(cls_repr)
        
        # Generar representación para respuesta a la declaración
        # Concatenar representación de sujeto y predicado
        combined_repr = torch.cat([enhanced_subject, enhanced_predicate], dim=1)
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