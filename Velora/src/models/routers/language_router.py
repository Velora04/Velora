"""
Enrutador Lingüístico para VELORA.

Este módulo implementa el enrutador específico para el dominio
lingüístico, que analiza y clasifica consultas de lenguaje natural.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ..components.attention import SelfAttentionBlock


class LanguageRouter(nn.Module):
    """
    Enrutador especializado para el dominio lingüístico.
    
    Identifica el tipo de consulta lingüística y analiza su
    estructura para asignarla al experto más apropiado.
    """
    
    def __init__(
        self,
        config,
        num_tasks: int = 3  # pregunta, comando, declaración
    ):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_tasks = num_tasks
        
        # Capas transformer para análisis contextual profundo
        self.transformer_layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=8,
                ff_dim=self.hidden_dim * 4,
                dropout_rate=config.dropout_rate,
                activation="gelu"
            ) for _ in range(3)  # 3 capas para análisis lingüístico
        ])
        
        # Clasificador de tipo de consulta
        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, num_tasks)
        )
        
        # Análisis de estructura lingüística
        self.structure_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 4)  # simple, compuesta, compleja, compuesta-compleja
        )
        
        # Estimador de confianza
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Detector de entidades
        self.entity_detector = EntityDetector(self.hidden_dim)
        
        # Detector de dominio temático
        self.topic_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 8)  # Categorías temáticas generales
        )
        
        # Matriz de compatibilidad entre tareas y expertos
        self.task_expert_compatibility = nn.Parameter(
            torch.ones(num_tasks, num_tasks)  # Inicialmente, mapeo uno-a-uno
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analiza entrada lingüística y determina el tipo de consulta.
        
        Args:
            hidden_states: Representación vectorial [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara para tokens de padding [batch_size, seq_len]
            
        Returns:
            Diccionario con información de enrutamiento lingüístico
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Procesar a través de las capas transformer
        sequence_repr = hidden_states
        for layer in self.transformer_layers:
            sequence_repr, _ = layer(sequence_repr, attention_mask)
        
        # Para clasificación, usar el primer token como representación [CLS]
        cls_repr = sequence_repr[:, 0]
        
        # Alternativa: ponderación atencional de todos los tokens
        # Esta implementación es más robusta para análisis lingüístico
        global_attention_weights = torch.softmax(
            torch.matmul(
                cls_repr.unsqueeze(1),  # [batch_size, 1, hidden_dim]
                sequence_repr.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
            ) / (self.hidden_dim ** 0.5),  # Escalado
            dim=-1
        )  # [batch_size, 1, seq_len]
        
        # Aplicar máscara si está disponible
        if attention_mask is not None:
            global_attention_weights = global_attention_weights * attention_mask.unsqueeze(1)
            global_attention_weights = global_attention_weights / global_attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        
        # Obtener representación ponderada atencional
        att_weighted_repr = torch.bmm(
            global_attention_weights,  # [batch_size, 1, seq_len]
            sequence_repr  # [batch_size, seq_len, hidden_dim]
        ).squeeze(1)  # [batch_size, hidden_dim]
        
        # Combinar representación CLS y ponderación atencional
        pooled_repr = (cls_repr + att_weighted_repr) / 2
        
        # Clasificar tipo de consulta
        task_logits = self.task_classifier(pooled_repr)
        task_probs = F.softmax(task_logits, dim=-1)
        task = torch.argmax(task_probs, dim=-1)
        
        # Analizar estructura lingüística
        structure_logits = self.structure_analyzer(pooled_repr)
        structure_probs = F.softmax(structure_logits, dim=-1)
        structure_type = torch.argmax(structure_probs, dim=-1)
        
        # Estimar confianza en la clasificación
        confidence = self.confidence_estimator(pooled_repr)
        
        # Detectar entidades
        entity_info = self.entity_detector(sequence_repr, attention_mask)
        
        # Clasificar dominio temático
        topic_logits = self.topic_classifier(pooled_repr)
        topic_probs = F.softmax(topic_logits, dim=-1)
        topic = torch.argmax(topic_probs, dim=-1)
        
        # Calcular compatibilidad con expertos
        # Normalizar matriz de compatibilidad
        compat_matrix = F.softmax(self.task_expert_compatibility, dim=-1)
        
        # Multiplicar probabilidades de tarea por matriz de compatibilidad
        expert_weights = torch.matmul(task_probs, compat_matrix)
        
        # Compilar resultados
        routing_info = {
            # Información de tarea
            'task': task,
            'task_logits': task_logits,
            'task_probs': task_probs,
            
            # Información de estructura
            'structure_type': structure_type,
            'structure_probs': structure_probs,
            
            # Información de entidades
            'has_entities': entity_info['has_entities'],
            'entity_positions': entity_info['entity_positions'],
            'entity_types': entity_info['entity_types'],
            
            # Información temática
            'topic': topic,
            'topic_probs': topic_probs,
            
            # Metainformación
            'confidence': confidence,
            'attention_weights': global_attention_weights,
            
            # Pesos para expertos
            'expert_weights': expert_weights,
            
            # Representación
            'sequence_representation': pooled_repr
        }
        
        return routing_info
    
    def explain_decisions(self, routing_info: Dict[str, torch.Tensor]) -> List[str]:
        """
        Genera explicaciones textuales de las decisiones de enrutamiento.
        
        Args:
            routing_info: Diccionario con información de enrutamiento
            
        Returns:
            Lista de explicaciones en formato legible
        """
        # Mapeos para interpretación
        task_names = ['pregunta', 'comando', 'declaración']
        structure_names = ['simple', 'compuesta', 'compleja', 'compuesta-compleja']
        topic_names = ['general', 'ciencia', 'tecnología', 'negocios', 
                       'arte', 'deportes', 'política', 'otros']
        
        # Extraer información clave
        batch_size = routing_info['task'].size(0)
        tasks = routing_info['task'].cpu().numpy()
        structures = routing_info['structure_type'].cpu().numpy()
        topics = routing_info['topic'].cpu().numpy()
        has_entities = routing_info['has_entities'].cpu().numpy()
        confidence_values = routing_info['confidence'].cpu().numpy()
        
        # Generar explicaciones
        explanations = []
        
        for i in range(batch_size):
            task_idx = tasks[i]
            task_name = task_names[task_idx]
            
            structure_idx = structures[i]
            structure_name = structure_names[structure_idx]
            
            topic_idx = topics[i]
            topic_name = topic_names[topic_idx]
            
            confidence = confidence_values[i][0]
            entities_present = "Sí" if has_entities[i] else "No"
            
            # Formatear explicación
            explanation = f"Tipo de consulta: {task_name} (confianza: {confidence:.2f}). "
            explanation += f"Estructura lingüística: {structure_name}. "
            explanation += f"Dominio temático: {topic_name}. "
            explanation += f"Presencia de entidades: {entities_present}."
            
            explanations.append(explanation)
        
        return explanations


class EntityDetector(nn.Module):
    """
    Detector de entidades nombradas para el enrutador lingüístico.
    
    Identifica la presencia y posición de entidades nombradas
    (personas, lugares, organizaciones, etc.) en el texto.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Detector de tokens de entidad
        self.entity_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 7)  # 6 tipos de entidad + no entidad
        )
        
        # Detector global de presencia de entidades
        self.global_entity_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detecta entidades en la representación de entrada.
        
        Args:
            hidden_states: Representación vectorial [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara para tokens de padding [batch_size, seq_len]
            
        Returns:
            Diccionario con información sobre entidades detectadas
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Clasificar tipo de entidad para cada token
        entity_logits = self.entity_classifier(hidden_states)  # [batch_size, seq_len, 7]
        entity_probs = F.softmax(entity_logits, dim=-1)
        
        # El último tipo (índice 6) es "no entidad"
        is_entity_prob = 1.0 - entity_probs[:, :, 6].unsqueeze(-1)  # [batch_size, seq_len, 1]
        entity_types = torch.argmax(entity_logits[:, :, :6], dim=-1)  # [batch_size, seq_len]
        
        # Aplicar máscara de atención si está disponible
        if attention_mask is not None:
            is_entity_prob = is_entity_prob * attention_mask.unsqueeze(-1)
        
        # Determinar posiciones con entidades (umbral de probabilidad 0.5)
        entity_positions = (is_entity_prob > 0.5).squeeze(-1)  # [batch_size, seq_len]
        
        # Detección global de presencia de entidades
        # Promediar representación de secuencia
        if attention_mask is not None:
            masked_states = hidden_states * attention_mask.unsqueeze(-1)
            seq_lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            avg_repr = masked_states.sum(dim=1) / seq_lengths
        else:
            avg_repr = hidden_states.mean(dim=1)
        
        # Detección global
        has_entities_prob = self.global_entity_detector(avg_repr)  # [batch_size, 1]
        has_entities = has_entities_prob > 0.5  # [batch_size, 1]
        
        return {
            'has_entities': has_entities,
            'entity_positions': entity_positions,
            'entity_types': entity_types,
            'entity_probs': is_entity_prob
        }