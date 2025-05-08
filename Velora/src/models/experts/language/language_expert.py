"""
Experto de lenguaje principal para VELORA.

Este módulo implementa el experto especializado en procesamiento
de lenguaje natural, coordinando los sub-expertos específicos.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from ..components.attention import SelfAttentionBlock
from .question_expert import QuestionExpert
from .command_expert import CommandExpert
from .statement_expert import StatementExpert


class LanguageExpert(nn.Module):
    """
    Experto de lenguaje para el sistema VELORA.
    
    Maneja el procesamiento de consultas lingüísticas mediante
    expertos especializados para diferentes tipos de consultas.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Encoder base para procesamiento de secuencias
        self.encoder_layers = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                ff_dim=self.hidden_dim * config.ff_expansion_factor,
                dropout_rate=config.dropout_rate,
                activation=config.activation_function
            ) for _ in range(4)  # 4 capas para análisis lingüístico profundo
        ])
        
        # Expertos específicos por tipo de consulta
        self.task_experts = nn.ModuleDict({
            'question': QuestionExpert(config),
            'command': CommandExpert(config),
            'statement': StatementExpert(config)
        })
        
        # Clasificador refinado de tipo de consulta
        self.task_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, 3)  # 3 tipos: pregunta, comando, declaración
        )
        
        # Estimador de confianza para decisiones
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Generador de representación final
        self.output_generator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_hint: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Procesa una consulta lingüística.
        
        Args:
            hidden_states: Representación de entrada [batch_size, seq_len, hidden_dim]
            task_hint: Indicación opcional de tipo de consulta (0=pregunta, 1=comando, 2=declaración)
            attention_mask: Máscara de atención para secuencia [batch_size, seq_len]
            
        Returns:
            Representación procesada y metadatos
        """
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        
        # Procesar a través de encoder base
        sequence_repr = hidden_states
        attentions = []
        
        for layer in self.encoder_layers:
            sequence_repr, attention = layer(sequence_repr, attention_mask, return_attention=True)
            if attention is not None:
                attentions.append(attention)
        
        # Obtener representación para clasificación
        # Usar token [CLS] (primer token)
        cls_repr = sequence_repr[:, 0]
        
        # Clasificar tipo de consulta si no se provee hint
        if task_hint is None:
            task_logits = self.task_classifier(cls_repr)
            task_probs = F.softmax(task_logits, dim=-1)
            task = torch.argmax(task_probs, dim=-1)
        else:
            # Usar el hint proporcionado
            if isinstance(task_hint, int):
                task = torch.tensor([task_hint] * batch_size, device=device)
            else:
                task = task_hint
            
            # Generar logits simulados para compatibilidad
            task_logits = F.one_hot(task, num_classes=3).float() * 10  # Alta confianza
            task_probs = F.softmax(task_logits, dim=-1)
        
        # Estimar confianza
        confidence = self.confidence_estimator(cls_repr)
        
        # Mapeo de índice de tarea a nombre
        task_mapping = {0: 'question', 1: 'command', 2: 'statement'}
        
        # Procesar con expertos según tipo de tarea
        expert_outputs = []
        expert_attentions = []
        
        for i in range(batch_size):
            task_idx = task[i].item()
            task_name = task_mapping[task_idx]
            expert = self.task_experts[task_name]
            
            # Extraer secuencia para esta muestra
            sample_sequence = sequence_repr[i:i+1]
            sample_mask = None if attention_mask is None else attention_mask[i:i+1]
            
            # Aplicar experto
            expert_output, expert_attention = expert(
                sample_sequence, 
                attention_mask=sample_mask, 
                return_attention=True
            )
            
            expert_outputs.append(expert_output)
            if expert_attention is not None:
                expert_attentions.append(expert_attention)
        
        # Combinar salidas de expertos
        combined_output = torch.cat(expert_outputs, dim=0)
        
        # Generar representación final
        final_output = self.output_generator(combined_output)
        
        # Mantener compatibilidad con formato de MathExpert
        pooled_output = final_output.mean(dim=1) if final_output.dim() > 2 else final_output
        
        # Compilar metadatos
        metadata = {
            'task': task,
            'task_logits': task_logits,
            'task_probs': task_probs,
            'confidence': confidence,
            'encoder_attentions': attentions,
            'expert_attentions': expert_attentions,
            'sequence_representation': sequence_repr,
            'cls_representation': cls_repr,
            'expert_output': combined_output,
            'pooled_output': pooled_output
        }
        
        return final_output, metadata
    
    def explain(self, metadata: Dict[str, torch.Tensor]) -> List[str]:
        """
        Genera explicaciones textuales del procesamiento lingüístico.
        
        Args:
            metadata: Metadatos del procesamiento
            
        Returns:
            Lista de explicaciones en formato legible
        """
        # Mapeo de índices a nombres de tarea
        task_names = ['pregunta', 'comando', 'declaración']
        
        # Extraer información relevante
        tasks = metadata['task'].cpu().numpy()
        confidence_values = metadata['confidence'].cpu().numpy()
        
        # Generar explicaciones
        explanations = []
        
        for i in range(len(tasks)):
            task_idx = tasks[i]
            task_name = task_names[task_idx]
            confidence = confidence_values[i][0]
            
            # Formatear explicación
            explanation = f"Tipo de consulta: {task_name}\n"
            explanation += f"Confianza: {confidence:.4f}\n"
            
            # Información específica según tipo
            if task_idx == 0:  # Pregunta
                explanation += "Procesado por experto de preguntas.\n"
                explanation += "Enfocado en extraer información relevante para respuesta."
            elif task_idx == 1:  # Comando
                explanation += "Procesado por experto de comandos.\n"
                explanation += "Orientado a entender acción solicitada y parámetros."
            else:  # Declaración
                explanation += "Procesado por experto de declaraciones.\n"
                explanation += "Centrado en análisis y evaluación de afirmaciones."
            
            explanations.append(explanation)
        
        return explanations