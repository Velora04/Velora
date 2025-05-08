"""
Enrutador Neural Primario para VELORA.

Este módulo implementa el enrutador principal que determina
la asignación de entradas a los expertos apropiados.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ..components.attention import SelfAttentionBlock


class PrimaryRouter(nn.Module):
    """
    Enrutador Neural Primario para VELORA.
    
    Determina el dominio de la entrada (aritmético, lenguaje)
    y asigna pesos a los expertos para procesar la consulta.
    """
    
    def __init__(
        self,
        config,
        num_domains: int = 2,  # Aritmético y lenguaje
    ):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_domains = num_domains
        
        # Encoder basado en transformer para análisis contextual
        self.encoder_layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                ff_dim=self.hidden_dim * config.ff_expansion_factor,
                dropout_rate=config.dropout_rate,
                activation=config.activation_function
            ) for _ in range(4)  # 4 capas para análisis profundo
        ])
        
        # Clasificador de dominio
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, num_domains)
        )
        
        # Estimador de confianza para decisiones de enrutamiento
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Clasificadores de tareas específicas de dominio
        
        # Clasificador para operaciones matemáticas
        self.math_operation_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 4)  # 4 operaciones: suma, resta, mult, div
        )
        
        # Clasificador para tareas lingüísticas
        self.language_task_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 3)  # 3 tipos: pregunta, comando, declaración
        )
        
        # Analizador de complejidad de tarea
        self.complexity_analyzer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.LayerNorm(self.hidden_dim // 4),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 4, 3)  # 3 niveles: simple, medio, complejo
        )
        
        # Parámetro para ajustar temperatura de softmax
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Analiza la entrada y determina su dominio y características.
        
        Args:
            hidden_states: Representación vectorial de la entrada [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara para tokens de padding [batch_size, seq_len]
            
        Returns:
            Diccionario con decisiones de enrutamiento y metadatos
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Procesar a través de las capas de transformer
        for layer in self.encoder_layers:
            hidden_states, _ = layer(hidden_states, attention_mask)
        
        # Obtener representación de secuencia completa (promedio de tokens)
        # Alternativa: usar el primer token como representación [CLS]
        # Crear máscara para evitar considerar padding en el promedio
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
            seq_lengths = torch.clamp(seq_lengths, min=1)  # Evitar división por cero
            pooled_output = sum_hidden / seq_lengths
        else:
            # Si no hay máscara, simplemente promediar
            pooled_output = hidden_states.mean(dim=1)
        
        # Clasificación de dominio
        domain_logits = self.domain_classifier(pooled_output)
        # Aplicar temperatura para controlar nitidez de la distribución
        scaled_logits = domain_logits / self.temperature
        domain_probs = F.softmax(scaled_logits, dim=-1)
        
        # Determinar dominio primario
        primary_domain = torch.argmax(domain_probs, dim=-1)
        
        # Estimación de confianza
        confidence = self.confidence_estimator(pooled_output)
        
        # Clasificación de tarea específica de dominio
        math_logits = self.math_operation_classifier(pooled_output)
        math_probs = F.softmax(math_logits, dim=-1)
        math_operation = torch.argmax(math_probs, dim=-1)
        
        language_logits = self.language_task_classifier(pooled_output)
        language_probs = F.softmax(language_logits, dim=-1)
        language_task = torch.argmax(language_probs, dim=-1)
        
        # Análisis de complejidad
        complexity_logits = self.complexity_analyzer(pooled_output)
        complexity_probs = F.softmax(complexity_logits, dim=-1)
        complexity_level = torch.argmax(complexity_probs, dim=-1)
        
        # Preparar pesos de expertos para fusión
        # Inicialmente basados en probabilidades de dominio
        expert_weights = domain_probs.clone()
        
        # Modificar pesos según confianza y complejidad
        # Si la confianza es baja, activar múltiples expertos
        confidence_threshold = 0.7
        low_confidence_mask = (confidence < confidence_threshold).float().squeeze(-1)
        
        # Para casos de baja confianza, suavizar la distribución
        # Para mantener expert_weights normalizado, usamos una interpolación
        uniform_weights = torch.ones_like(expert_weights) / self.num_domains
        expert_weights = (1 - low_confidence_mask).unsqueeze(-1) * expert_weights + \
                         low_confidence_mask.unsqueeze(-1) * uniform_weights
        
        # Compilar resultados
        routing_info = {
            # Información de dominio
            'primary_domain': primary_domain,
            'domain_logits': domain_logits,
            'domain_probs': domain_probs,
            
            # Información de confianza
            'confidence': confidence,
            
            # Información de tarea
            'math_operation': math_operation,
            'math_logits': math_logits,
            'math_probs': math_probs,
            'language_task': language_task,
            'language_logits': language_logits,
            'language_probs': language_probs,
            
            # Información de complejidad
            'complexity_level': complexity_level,
            'complexity_probs': complexity_probs,
            
            # Pesos para expertos
            'expert_weights': expert_weights,
            
            # Metadatos adicionales
            'temperature': self.temperature,
            'pooled_representation': pooled_output
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
        domain_names = ['aritmético', 'lenguaje']
        math_op_names = ['suma', 'resta', 'multiplicación', 'división']
        language_task_names = ['pregunta', 'comando', 'declaración']
        complexity_names = ['simple', 'moderada', 'compleja']
        
        # Extraer información clave
        batch_size = routing_info['primary_domain'].size(0)
        primary_domains = routing_info['primary_domain'].cpu().numpy()
        confidence_values = routing_info['confidence'].cpu().numpy()
        math_operations = routing_info['math_operation'].cpu().numpy()
        language_tasks = routing_info['language_task'].cpu().numpy()
        complexity_levels = routing_info['complexity_level'].cpu().numpy()
        
        # Generar explicaciones para cada elemento del batch
        explanations = []
        
        for i in range(batch_size):
            domain_idx = primary_domains[i]
            domain_name = domain_names[domain_idx]
            
            confidence = confidence_values[i][0]  # Extraer escalar
            
            complexity = complexity_names[complexity_levels[i]]
            
            # Base de la explicación
            explanation = f"Dominio detectado: {domain_name} (confianza: {confidence:.2f}). "
            explanation += f"Tarea de complejidad {complexity}. "
            
            # Añadir detalles específicos del dominio
            if domain_idx == 0:  # Aritmético
                math_op = math_op_names[math_operations[i]]
                explanation += f"Operación matemática: {math_op}."
            else:  # Lenguaje
                lang_task = language_task_names[language_tasks[i]]
                explanation += f"Tipo de consulta: {lang_task}."
            
            explanations.append(explanation)
        
        return explanations