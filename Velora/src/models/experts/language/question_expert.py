"""
Experto de preguntas para VELORA.

Este módulo implementa el experto especializado en procesamiento
de consultas en forma de pregunta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ...components.attention import SelfAttentionBlock


class QuestionExpert(nn.Module):
    """
    Experto especializado en procesamiento de preguntas.
    
    Implementa capacidades específicas para entender, analizar
    y procesar consultas en forma interrogativa.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Capas transformer especializadas para preguntas
        self.transformer_layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                ff_dim=self.hidden_dim * config.ff_expansion_factor,
                dropout_rate=config.dropout_rate,
                activation=config.activation_function
            ) for _ in range(3)  # 3 capas especializadas
        ])
        
        # Clasificador de tipo de pregunta
        self.question_type_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 7)  # Tipos: qué, quién, cuándo, dónde, cómo, por qué, otros
        )
        
        # Detector de palabras de pregunta
        self.wh_word_detector = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Extractor de tópico de pregunta (sobre qué se pregunta)
        self.topic_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Estimador de complejidad de la pregunta
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # 0: simple, 1: compleja
        )
        
        # Generador de representación de respuesta
        self.answer_generator = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # Concatena repr. pregunta y tópico
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
        Procesa una pregunta y genera representación orientada a respuesta.
        
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
        
        # Clasificar tipo de pregunta
        # Usar primer token [CLS] para clasificación
        cls_repr = sequence_repr[:, 0]
        question_type_logits = self.question_type_classifier(cls_repr)
        question_type_probs = F.softmax(question_type_logits, dim=-1)
        question_type = torch.argmax(question_type_probs, dim=-1)
        
        # Detectar palabras de pregunta (qué, quién, etc.)
        # Aplicar detector a cada token y obtener máximo
        wh_scores = self.wh_word_detector(sequence_repr)  # [batch_size, seq_len, 1]
        
        # Aplicar máscara de atención si está disponible
        if attention_mask is not None:
            wh_scores = wh_scores * attention_mask.unsqueeze(-1)
        
        # Encontrar posición con mayor probabilidad de ser palabra de pregunta
        wh_position = torch.argmax(wh_scores.squeeze(-1), dim=-1)  # [batch_size]
        
        # Extraer representación de tópico de la pregunta
        # Usar atención ponderada por palabras no-pregunta
        # Invertir scores de palabras de pregunta
        topic_weights = 1.0 - wh_scores
        
        # Normalizar pesos
        if attention_mask is not None:
            topic_weights = topic_weights * attention_mask.unsqueeze(-1)
        
        topic_weights = topic_weights / (topic_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Aplicar atención ponderada
        topic_repr = torch.bmm(
            topic_weights.transpose(1, 2),  # [batch_size, 1, seq_len]
            sequence_repr  # [batch_size, seq_len, hidden_dim]
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        # Mejorar representación de tópico
        enhanced_topic = self.topic_extractor(topic_repr)
        
        # Estimar complejidad
        complexity = self.complexity_estimator(cls_repr)
        
        # Generar representación para respuesta
        # Concatenar representación de pregunta y tópico
        combined_repr = torch.cat([cls_repr, enhanced_topic], dim=1)
        answer_repr = self.answer_generator(combined_repr)
        
        # Expandir a formato de secuencia si es necesario para compatibilidad
        # con interfaz de otros expertos
        if original_repr.dim() > 2:
            answer_repr = answer_repr.unsqueeze(1)
        
        # Preparar salida de atención si se solicita
        final_attention = None
        if return_attention and attentions:
            # Devolver última capa de atención
            final_attention = attentions[-1]
        
        return answer_repr, final_attention