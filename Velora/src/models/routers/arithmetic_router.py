"""
Enrutador Aritmético para VELORA.

Este módulo implementa el enrutador específico para el dominio
aritmético, que analiza y clasifica operaciones matemáticas.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from typing import Dict, List, Optional, Tuple, Any


class ArithmeticRouter(nn.Module):
    """
    Enrutador especializado para el dominio aritmético.
    
    Identifica el tipo de operación matemática y extrae
    operandos para direccionar la consulta al experto adecuado.
    """
    
    def __init__(
        self,
        config,
        num_operations: int = 4  # suma, resta, multiplicación, división
    ):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_operations = num_operations
        
        # MLP para clasificación de operación
        self.operation_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim // 2, num_operations)
        )
        
        # Red para extracción de operandos
        self.operand_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(self.hidden_dim, 2)  # Extraer 2 operandos para operaciones binarias
        )
        
        # Detector de patrones numéricos para mejorar extracción
        self.number_detector = NumberDetectorNetwork(self.hidden_dim)
        
        # Estimador de confianza específico
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Detector de complejidad computacional
        self.complexity_estimator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 1),
            nn.Sigmoid()  # 0: simple, 1: complejo
        )
        
        # Matriz de compatibilidad entre operaciones y expertos
        # Esta matriz se aprende durante el entrenamiento
        self.operation_expert_compatibility = nn.Parameter(
            torch.ones(num_operations, num_operations)  # Inicialmente, mapeo uno-a-uno
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None  # Útil para análisis simbólico
    ) -> Dict[str, torch.Tensor]:
        """
        Analiza entrada matemática y determina operación y operandos.
        
        Args:
            hidden_states: Representación vectorial [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara para tokens de padding [batch_size, seq_len]
            token_ids: IDs de tokens para análisis simbólico [batch_size, seq_len]
            
        Returns:
            Diccionario con información de enrutamiento aritmético
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Obtener representación de secuencia completa
        if attention_mask is not None:
            # Crear máscara para promediar tokens no enmascarados
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True)
            seq_lengths = torch.clamp(seq_lengths, min=1)  # Evitar división por cero
            sequence_repr = sum_hidden / seq_lengths
        else:
            sequence_repr = hidden_states.mean(dim=1)
        
        # Clasificar tipo de operación
        operation_logits = self.operation_classifier(sequence_repr)
        operation_probs = F.softmax(operation_logits, dim=-1)
        operation = torch.argmax(operation_probs, dim=-1)
        
        # Extraer operandos
        operands = self.operand_extractor(sequence_repr)
        
        # Aplicar detector de números para mejorar precisión
        number_detection_info = self.number_detector(hidden_states, attention_mask, token_ids)
        
        # Si el detector de números encuentra operandos con alta confianza, usarlos
        if number_detection_info['confidence'] > 0.8:
            detected_operands = number_detection_info['operands']
            # Mezclamos los operandos detectados y los predichos según confianza
            blend_factor = number_detection_info['confidence']
            operands = blend_factor * detected_operands + (1 - blend_factor) * operands
        
        # Estimar confianza en la clasificación
        confidence = self.confidence_estimator(sequence_repr)
        
        # Estimar complejidad computacional
        complexity = self.complexity_estimator(sequence_repr)
        
        # Calcular compatibilidad con expertos
        # Normalizar matriz de compatibilidad
        compat_matrix = F.softmax(self.operation_expert_compatibility, dim=-1)
        
        # Multiplicar probabilidades de operación por matriz de compatibilidad
        expert_weights = torch.matmul(operation_probs, compat_matrix)
        
        # Compilar resultados
        routing_info = {
            # Información de operación
            'operation': operation,
            'operation_logits': operation_logits,
            'operation_probs': operation_probs,
            
            # Información de operandos
            'operands': operands,
            'detected_operands': number_detection_info['operands'],
            'detection_confidence': number_detection_info['confidence'],
            
            # Metainformación
            'confidence': confidence,
            'complexity': complexity,
            
            # Pesos para expertos
            'expert_weights': expert_weights,
            
            # Representación
            'sequence_representation': sequence_repr
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
        operation_names = ['suma', 'resta', 'multiplicación', 'división']
        
        # Extraer información clave
        batch_size = routing_info['operation'].size(0)
        operations = routing_info['operation'].cpu().numpy()
        operands = routing_info['operands'].cpu().numpy()
        confidence_values = routing_info['confidence'].cpu().numpy()
        complexity_values = routing_info['complexity'].cpu().numpy()
        
        # Generar explicaciones
        explanations = []
        
        for i in range(batch_size):
            op_idx = operations[i]
            op_name = operation_names[op_idx]
            
            operand1, operand2 = operands[i]
            confidence = confidence_values[i][0]
            complexity = complexity_values[i][0]
            
            # Formatear explicación
            explanation = f"Operación detectada: {op_name} (confianza: {confidence:.2f}). "
            explanation += f"Operandos aproximados: {operand1:.2f} y {operand2:.2f}. "
            explanation += f"Complejidad computacional: {complexity:.2f}."
            
            explanations.append(explanation)
        
        return explanations


class NumberDetectorNetwork(nn.Module):
    """
    Red especializada para detectar números en representaciones vectoriales.
    
    Combina análisis neural con heurísticas para mejorar la extracción
    de operandos numéricos de la entrada.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Detector de dígitos por token
        self.digit_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Clasificador de posición decimal
        self.decimal_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Red para combinar dígitos en números completos
        self.number_aggregator = nn.GRU(
            input_size=hidden_dim + 2,  # hidden_dim + digit_score + decimal_score
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Proyección final para obtener valor numérico
        self.value_projector = nn.Linear(hidden_dim // 2, 1)
        
        # Estimador de confianza global
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Detecta números en la representación de entrada.
        
        Args:
            hidden_states: Representación vectorial [batch_size, seq_len, hidden_dim]
            attention_mask: Máscara para tokens de padding [batch_size, seq_len]
            token_ids: IDs de tokens para análisis simbólico [batch_size, seq_len]
            
        Returns:
            Diccionario con operandos detectados y confianza
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # Detectar probabilidad de que cada token sea un dígito
        digit_scores = self.digit_detector(hidden_states)  # [batch_size, seq_len, 1]
        
        # Detectar probabilidad de que cada token sea parte de posición decimal
        decimal_scores = self.decimal_classifier(hidden_states)  # [batch_size, seq_len, 1]
        
        # Combinar representaciones con scores
        combined_repr = torch.cat([
            hidden_states,
            digit_scores,
            decimal_scores
        ], dim=-1)  # [batch_size, seq_len, hidden_dim + 2]
        
        # Procesar secuencia para agregar dígitos en números
        aggregated_repr, last_hidden = self.number_aggregator(combined_repr)
        last_hidden = last_hidden.squeeze(0)  # [batch_size, hidden_dim // 2]
        
        # Estimar confianza global en la detección
        confidence = self.confidence_estimator(last_hidden)
        
        # En lugar de intentar una detección end-to-end compleja,
        # simplificamos extrayendo dos valores principales
        
        # Proyectar a valores numéricos (simplificado)
        value = self.value_projector(last_hidden)  # [batch_size, 1]
        
        # Segunda aproximación: dividir secuencia en dos partes y buscar un valor en cada una
        mid_point = seq_len // 2
        
        # Primera mitad
        if attention_mask is not None:
            first_half_mask = attention_mask[:, :mid_point]
            first_valid_indices = first_half_mask.sum(dim=1).clamp(min=1)
            first_half_repr = aggregated_repr[:, :mid_point]
            first_half_sum = torch.sum(first_half_repr * first_half_mask.unsqueeze(-1), dim=1)
            first_half_avg = first_half_sum / first_valid_indices.unsqueeze(-1)
        else:
            first_half_avg = aggregated_repr[:, :mid_point].mean(dim=1)
        
        # Segunda mitad
        if attention_mask is not None:
            second_half_mask = attention_mask[:, mid_point:]
            second_valid_indices = second_half_mask.sum(dim=1).clamp(min=1)
            second_half_repr = aggregated_repr[:, mid_point:]
            second_half_sum = torch.sum(second_half_repr * second_half_mask.unsqueeze(-1), dim=1)
            second_half_avg = second_half_sum / second_valid_indices.unsqueeze(-1)
        else:
            second_half_avg = aggregated_repr[:, mid_point:].mean(dim=1)
        
        # Proyectar a valores numéricos
        first_value = self.value_projector(first_half_avg)
        second_value = self.value_projector(second_half_avg)
        
        # Combinar en tensor de operandos
        operands = torch.cat([first_value, second_value], dim=1)  # [batch_size, 2]
        
        # Si se proporcionan token_ids, intentar análisis simbólico
        symbolic_operands = None
        if token_ids is not None:
            symbolic_operands = self._symbolic_number_extraction(token_ids)
            
            # Si la extracción simbólica tiene éxito, mezclar con resultados neurales
            if symbolic_operands is not None:
                # Aumentar confianza y mezclar resultados
                blend_factor = 0.7  # Peso para extracción simbólica
                operands = blend_factor * symbolic_operands + (1 - blend_factor) * operands
                confidence = torch.max(confidence, torch.tensor(0.8, device=confidence.device))
        
        return {
            'operands': operands,
            'confidence': confidence,
            'digit_scores': digit_scores,
            'decimal_scores': decimal_scores
        }
    
    def _symbolic_number_extraction(self, token_ids: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Intenta extraer operandos numérricos mediante análisis simbólico.
        
        Esta es una implementación simplificada que asumiría que tenemos
        una función para decodificar tokens a texto y luego analizar números.
        En un sistema real, se conectaría con el tokenizador.
        
        Args:
            token_ids: IDs de tokens [batch_size, seq_len]
            
        Returns:
            Tensor de operandos extraídos o None si falla
        """
        # Esta es una implementación simulada
        # En un sistema real, decodificaríamos los tokens y aplicaríamos
        # expresiones regulares u otras técnicas para extraer números
        
        batch_size = token_ids.size(0)
        device = token_ids.device
        
        # Devolver valores aleatorios para simular éxito de extracción
        # Con probabilidad de 0.5 para ilustrar el concepto
        if torch.rand(1).item() > 0.5:
            return torch.randn(batch_size, 2, device=device)
        else:
            return None