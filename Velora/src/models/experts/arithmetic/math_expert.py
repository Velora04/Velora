"""
Experto matemático principal para VELORA.

Este módulo implementa el experto especializado en operaciones
matemáticas, coordinando los sub-expertos para cada operación.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from .add_expert import AddExpert
from .subtract_expert import SubtractExpert
from .multiply_expert import MultiplyExpert
from .divide_expert import DivideExpert


class MathExpert(nn.Module):
    """
    Experto matemático para el sistema VELORA.
    
    Coordina expertos específicos para cada operación
    aritmética básica y proporciona capacidades de
    verificación y explicación.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Encoder común para representaciones matemáticas
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Expertos para operaciones específicas
        self.operation_experts = nn.ModuleDict({
            'add': AddExpert(config),
            'subtract': SubtractExpert(config),
            'multiply': MultiplyExpert(config),
            'divide': DivideExpert(config)
        })
        
        # Extractor de operandos
        self.operand_extractor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 2)  # Extraer 2 operandos
        )
        
        # Red de verificación simbólica
        self.symbolic_verifier = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, self.hidden_dim // 2),  # +3 para resultado y operación
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Red de generación de explicación
        self.explanation_generator = nn.Sequential(
            nn.Linear(self.hidden_dim + 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Capa de proyección para resultado final
        self.result_projector = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        operation_hint: Optional[int] = None,
        operands: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Procesa una consulta matemática.
        
        Args:
            hidden_states: Representación de entrada [batch_size, seq_len, hidden_dim]
            operation_hint: Indicación opcional de operación (0=suma, 1=resta, 2=mult, 3=div)
            operands: Operandos precalculados (opcional) [batch_size, 2]
            attention_mask: Máscara de atención para ignorar padding [batch_size, seq_len]
            
        Returns:
            Representación del resultado y metadatos
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Obtener representación de secuencia completa
        if hidden_states.dim() == 3:  # [batch_size, seq_len, hidden_dim]
            if attention_mask is not None:
                # Crear máscara expandida
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                # Suma ponderada
                sum_repr = torch.sum(hidden_states * mask_expanded, dim=1)
                # Normalizar por longitud de secuencia
                seq_lengths = torch.sum(attention_mask, dim=1, keepdim=True).clamp(min=1)
                sequence_repr = sum_repr / seq_lengths
            else:
                sequence_repr = hidden_states.mean(dim=1)
        else:  # Ya es [batch_size, hidden_dim]
            sequence_repr = hidden_states
        
        # Codificar representación matemática
        math_repr = self.encoder(sequence_repr)
        
        # Extraer operandos si no se proporcionan
        if operands is None:
            operands = self.operand_extractor(math_repr)
        
        # Mapeo de índice de operación a nombre
        op_mapping = {0: 'add', 1: 'subtract', 2: 'multiply', 3: 'divide'}
        
        # Si no se proporciona hint de operación, tratar de inferirla
        if operation_hint is None:
            # Inferir basado en características de la representación
            # Estrategia simple: probar todas las operaciones y seleccionar la de menor pérdida
            
            min_loss = float('inf')
            best_op = 0
            best_neural_result = None
            best_symbolic_result = None
            
            # Probar cada operación
            for op_idx, op_name in op_mapping.items():
                expert = self.operation_experts[op_name]
                neural_result, symbolic_result = expert(operands)
                
                # Calcular pérdida (aquí simplificada)
                # En un sistema real, esto podría involucrar un modelo de lenguaje
                # para evaluar cuán probable es la operación dada la entrada
                representational_loss = F.mse_loss(
                    neural_result, torch.zeros_like(neural_result)
                )
                
                if representational_loss.item() < min_loss:
                    min_loss = representational_loss.item()
                    best_op = op_idx
                    best_neural_result = neural_result
                    best_symbolic_result = symbolic_result
            
            # Usar la mejor operación inferida
            operation = torch.tensor([best_op] * batch_size, device=device)
            neural_result = best_neural_result
            symbolic_result = best_symbolic_result
            
        else:
            # Convertir hint a tensor si es escalar
            if isinstance(operation_hint, int):
                operation = torch.tensor([operation_hint] * batch_size, device=device)
            else:
                operation = operation_hint
            
            # Procesar con el experto correspondiente
            results_neural = []
            results_symbolic = []
            
            for i in range(batch_size):
                op_idx = operation[i].item()
                op_name = op_mapping[op_idx]
                expert = self.operation_experts[op_name]
                
                # Extraer operandos para esta muestra
                sample_operands = operands[i].unsqueeze(0)
                
                # Aplicar experto
                n_result, s_result = expert(sample_operands)
                results_neural.append(n_result)
                results_symbolic.append(s_result)
            
            # Combinar resultados
            neural_result = torch.cat(results_neural, dim=0)
            symbolic_result = torch.cat(results_symbolic, dim=0)
        
        # Verificación simbólica 
        # Concatenamos representación, resultado y operación
        operation_expanded = F.one_hot(operation, num_classes=4).float()
        verification_input = torch.cat([
            math_repr,
            neural_result.unsqueeze(1),
            symbolic_result.unsqueeze(1),
            operation_expanded
        ], dim=1)
        
        verification_score = self.symbolic_verifier(verification_input)
        
        # Si hay discrepancia significativa entre resultados neural y simbólico,
        # priorizar el simbólico que es más preciso
        result_discrepancy = torch.abs(neural_result - symbolic_result)
        use_symbolic = (result_discrepancy > 0.1) | (verification_score < 0.8)
        
        # Resultado final como interpolación entre neural y simbólico
        symbolic_weight = use_symbolic.float().unsqueeze(1)
        final_result = (1 - symbolic_weight) * neural_result + symbolic_weight * symbolic_result
        
        # Proyectar a espacio de representación
        final_repr = math_repr.clone()
        # Incluir información del resultado en la representación
        final_repr = self.result_projector(final_repr + final_result.unsqueeze(1))
        
        # Generar información para explicación
        explanation_input = torch.cat([
            math_repr,
            final_result.unsqueeze(1),
            operation_expanded
        ], dim=1)
        explanation_repr = self.explanation_generator(explanation_input)
        
        # Compilar metadatos
        metadata = {
            'operation': operation,
            'operands': operands,
            'neural_result': neural_result,
            'symbolic_result': symbolic_result,
            'verification_score': verification_score,
            'final_result': final_result,
            'explanation_repr': explanation_repr
        }
        
        return final_repr, metadata
    
    def explain(self, metadata: Dict[str, torch.Tensor]) -> List[str]:
        """
        Genera explicaciones textuales del procesamiento matemático.
        
        Args:
            metadata: Metadatos del procesamiento
            
        Returns:
            Lista de explicaciones en formato legible
        """
        # Mapeo de índices a nombres de operación
        op_names = ['suma', 'resta', 'multiplicación', 'división']
        
        # Extraer información relevante
        operations = metadata['operation'].cpu().numpy()
        operands = metadata['operands'].cpu().numpy()
        final_results = metadata['final_result'].cpu().numpy()
        verification_scores = metadata['verification_score'].cpu().numpy()
        
        # Generar explicaciones
        explanations = []
        
        for i in range(len(operations)):
            op_idx = operations[i]
            op_name = op_names[op_idx]
            
            a, b = operands[i]
            result = final_results[i]
            verification = verification_scores[i][0]
            
            # Formatear explicación
            explanation = f"Operación: {op_name}\n"
            explanation += f"Operandos: {a:.4f} y {b:.4f}\n"
            explanation += f"Resultado: {result:.4f}\n"
            explanation += f"Confianza: {verification:.4f}"
            
            # Añadir notas específicas para división por cero
            if op_idx == 3 and abs(b) < 1e-6:
                explanation += "\nNota: División con denominador cercano a cero."
            
            explanations.append(explanation)
        
        return explanations