"""
Funciones de pérdida para entrenamiento integrado de VELORA.

Este módulo implementa las funciones de pérdida compuestas
que coordinan los diferentes objetivos de entrenamiento.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


class VeloraLoss(nn.Module):
    """
    Función de pérdida compuesta para VELORA.
    
    Combina múltiples objetivos de entrenamiento:
    - Clasificación de dominio
    - Clasificación de tareas específicas por dominio
    - Precisión de resultados
    - Consistencia entre componentes
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Pesos para diferentes componentes de pérdida
        self.domain_weight = getattr(config, 'domain_loss_weight', 1.0)
        self.tasks_weight = getattr(config, 'tasks_loss_weight', 1.0)
        self.results_weight = getattr(config, 'results_loss_weight', 1.0)
        self.consistency_weight = getattr(config, 'consistency_loss_weight', 0.5)
        self.memory_weight = getattr(config, 'memory_loss_weight', 0.2)
        
        # Funciones de pérdida para diferentes componentes
        self.domain_criterion = nn.CrossEntropyLoss()
        self.math_op_criterion = nn.CrossEntropyLoss()
        self.language_task_criterion = nn.CrossEntropyLoss()
        self.result_criterion = nn.MSELoss()
        
        # Para resultados numéricos, también considerar error relativo
        self.use_relative_error = getattr(config, 'use_relative_error', True)
        self.relative_error_weight = getattr(config, 'relative_error_weight', 0.5)
    
    def forward(
        self,
        outputs: torch.Tensor,
        metadata: Dict[str, Any],
        domain_labels: torch.Tensor,
        math_op_labels: torch.Tensor,
        language_task_labels: torch.Tensor,
        result_labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calcula la pérdida compuesta.
        
        Args:
            outputs: Salida del modelo [batch_size, hidden_dim]
            metadata: Metadatos del forward pass
            domain_labels: Etiquetas de dominio [batch_size]
            math_op_labels: Etiquetas de operación matemática [batch_size]
            language_task_labels: Etiquetas de tarea lingüística [batch_size]
            result_labels: Etiquetas de resultado [batch_size]
            
        Returns:
            Diccionario con pérdidas desglosadas y pérdida total
        """
        batch_size = outputs.size(0)
        device = outputs.device
        
        # Extraer información relevante de los metadatos
        primary_routing = metadata['primary_routing']
        
        # 1. Pérdida de dominio
        domain_logits = primary_routing['domain_logits']
        domain_loss = self.domain_criterion(domain_logits, domain_labels)
        
        # 2. Pérdidas específicas por dominio
        tasks_loss = torch.tensor(0.0, device=device)
        
        # Crear máscaras para cada dominio
        math_mask = (domain_labels == 0)
        lang_mask = (domain_labels == 1)
        
        # 2.1 Pérdida para tareas matemáticas
        math_samples = math_mask.sum().item()
        if math_samples > 0:
            # Si hay muestras matemáticas, calcular pérdida de operación
            math_op_logits = primary_routing['math_logits']
            math_op_loss = self.math_op_criterion(
                math_op_logits[math_mask],
                math_op_labels[math_mask]
            )
            tasks_loss = tasks_loss + math_op_loss
        
        # 2.2 Pérdida para tareas lingüísticas
        lang_samples = lang_mask.sum().item()
        if lang_samples > 0:
            # Si hay muestras lingüísticas, calcular pérdida de tarea
            lang_task_logits = primary_routing['language_logits']
            lang_task_loss = self.language_task_criterion(
                lang_task_logits[lang_mask],
                language_task_labels[lang_mask]
            )
            tasks_loss = tasks_loss + lang_task_loss
        
        # 3. Pérdida para resultados
        # Extraer resultados numéricos para muestras matemáticas
        results_loss = torch.tensor(0.0, device=device)
        
        if 'expert_metadata' in metadata and math_samples > 0:
            # Extraer resultados finales
            math_results = []
            
            for i in range(batch_size):
                if math_mask[i]:
                    if 'final_result' in metadata['expert_metadata'][i]:
                        math_results.append(metadata['expert_metadata'][i]['final_result'])
                    elif 'neural_result' in metadata['expert_metadata'][i]:
                        # Alternativa: usar resultado neural si no hay final
                        math_results.append(metadata['expert_metadata'][i]['neural_result'])
            
            if math_results:
                # Convertir lista a tensor
                math_results_tensor = torch.stack(math_results)
                
                # Extraer etiquetas de resultado para muestras matemáticas
                math_result_labels = result_labels[math_mask]
                
                # Calcular error cuadrático medio
                mse_loss = self.result_criterion(math_results_tensor, math_result_labels)
                results_loss = results_loss + mse_loss
                
                # Opcionalmente, añadir error relativo
                if self.use_relative_error:
                    # Evitar división por cero
                    epsilon = 1e-8
                    abs_labels = torch.abs(math_result_labels) + epsilon
                    
                    # Calcular error relativo
                    rel_error = torch.abs(math_results_tensor - math_result_labels) / abs_labels
                    rel_loss = torch.mean(rel_error)
                    
                    # Añadir a pérdida de resultados
                    results_loss = results_loss + self.relative_error_weight * rel_loss
        
        # 4. Pérdida de consistencia
        consistency_loss = torch.tensor(0.0, device=device)
        
        if 'fusion' in metadata and 'consistency_score' in metadata['fusion']:
            # Promover alta consistencia (cerca de 1.0)
            consistency_scores = metadata['fusion']['consistency_score']
            consistency_loss = torch.mean(1.0 - consistency_scores)
        
        # 5. Pérdida de memoria (opcional)
        memory_loss = torch.tensor(0.0, device=device)
        
        if 'memory' in metadata and 'write_gate' in metadata['memory']:
            # Regularizar uso de memoria: ni demasiado ni demasiado poco
            write_gate = metadata['memory']['write_gate']
            memory_usage = metadata['memory']['memory_usage']
            
            # Penalizar valores extremos (0 o 1)
            gate_reg = torch.mean((write_gate - 0.5).pow(2))
            
            # Promover uso balanceado de memoria (ni muy poco ni demasiado)
            usage_reg = torch.mean((memory_usage - 0.5).pow(2))
            
            memory_loss = gate_reg + usage_reg
        
        # Combinar todas las pérdidas con sus pesos
        total_loss = (
            self.domain_weight * domain_loss +
            self.tasks_weight * tasks_loss +
            self.results_weight * results_loss +
            self.consistency_weight * consistency_loss +
            self.memory_weight * memory_loss
        )
        
        # Devolver pérdidas desglosadas y total
        return {
            'domain': domain_loss.item(),
            'tasks': tasks_loss.item(),
            'results': results_loss.item(),
            'consistency': consistency_loss.item(),
            'memory': memory_loss.item(),
            'total': total_loss
        }


class DomainClassificationLoss(nn.Module):
    """
    Función de pérdida específica para clasificación de dominio.
    
    Incluye regularización y calibración de confianza.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        
        # Factor para regularización
        self.confidence_reg_weight = getattr(config, 'confidence_reg_weight', 0.1)
        
        # Temperatura para calibración
        self.temperature = getattr(config, 'temperature', 1.0)
    
    def forward(
        self,
        domain_logits: torch.Tensor,
        domain_labels: torch.Tensor,
        confidence: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Calcula pérdida de clasificación de dominio.
        
        Args:
            domain_logits: Logits para clasificación de dominio [batch_size, num_domains]
            domain_labels: Etiquetas de dominio [batch_size]
            confidence: Estimaciones de confianza [batch_size, 1]
            
        Returns:
            Pérdida total de clasificación de dominio
        """
        # Aplicar temperatura para calibración
        scaled_logits = domain_logits / self.temperature
        
        # Calcular pérdida de clasificación
        ce_loss = self.criterion(scaled_logits, domain_labels)
        
        # Regularización de confianza (si se proporciona)
        conf_loss = torch.tensor(0.0, device=domain_logits.device)
        
        if confidence is not None:
            # Extraer confianza para muestras correctamente clasificadas
            pred_domains = torch.argmax(domain_logits, dim=1)
            correct_mask = (pred_domains == domain_labels)
            
            if correct_mask.sum() > 0:
                correct_conf = confidence[correct_mask]
                
                # Penalizar baja confianza en predicciones correctas
                correct_conf_loss = torch.mean(1.0 - correct_conf)
                
                # Penalizar alta confianza en predicciones incorrectas
                if (~correct_mask).sum() > 0:
                    incorrect_conf = confidence[~correct_mask]
                    incorrect_conf_loss = torch.mean(incorrect_conf)
                else:
                    incorrect_conf_loss = torch.tensor(0.0, device=domain_logits.device)
                
                # Combinar
                conf_loss = correct_conf_loss + incorrect_conf_loss
        
        # Pérdida total
        total_loss = ce_loss + self.confidence_reg_weight * conf_loss
        
        return total_loss


class TaskClassificationLoss(nn.Module):
    """
    Función de pérdida para clasificación de tareas específicas de dominio.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.math_criterion = nn.CrossEntropyLoss()
        self.language_criterion = nn.CrossEntropyLoss()
    
    def forward(
        self,
        math_logits: torch.Tensor,
        math_labels: torch.Tensor,
        math_mask: torch.Tensor,
        language_logits: torch.Tensor,
        language_labels: torch.Tensor,
        language_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcula pérdida de clasificación de tareas específicas.
        
        Args:
            math_logits: Logits para operaciones matemáticas [batch_size, num_operations]
            math_labels: Etiquetas de operación [batch_size]
            math_mask: Máscara para muestras matemáticas [batch_size]
            language_logits: Logits para tareas lingüísticas [batch_size, num_tasks]
            language_labels: Etiquetas de tarea lingüística [batch_size]
            language_mask: Máscara para muestras lingüísticas [batch_size]
            
        Returns:
            Pérdida total de clasificación de tareas
        """
        device = math_logits.device
        loss = torch.tensor(0.0, device=device)
        
        # Pérdida para tareas matemáticas
        math_samples = math_mask.sum().item()
        if math_samples > 0:
            math_loss = self.math_criterion(
                math_logits[math_mask],
                math_labels[math_mask]
            )
            loss = loss + math_loss
        
        # Pérdida para tareas lingüísticas
        language_samples = language_mask.sum().item()
        if language_samples > 0:
            language_loss = self.language_criterion(
                language_logits[language_mask],
                language_labels[language_mask]
            )
            loss = loss + language_loss
        
        return loss