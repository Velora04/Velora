"""
Modelo VELORA completo.

Este módulo implementa la arquitectura principal de VELORA,
integrando enrutadores, expertos y mecanismos de fusión.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any

from .routers.primary_router import PrimaryRouter
from .routers.arithmetic_router import ArithmeticRouter
from .routers.language_router import LanguageRouter
from .experts.arithmetic.math_expert import MathExpert
from .experts.language.language_expert import LanguageExpert
from .components.memory import WorkingMemory, ContextManager
from .integration.fusion import ExpertFusion


class VELORA(nn.Module):
    """
    VELORA: Versatile Expert Learning for Operational Reasoning and Analysis.
    
    Arquitectura neuronal modular que combina múltiples expertos especializados
    coordinados por un sistema inteligente de enrutamiento.
    """
    
    def __init__(
        self,
        config,
        math_expert: Optional[MathExpert] = None,
        language_expert: Optional[LanguageExpert] = None,
        freeze_experts: bool = False
    ):
        """
        Inicializa el modelo VELORA.
        
        Args:
            config: Configuración del modelo
            math_expert: Experto matemático pre-entrenado (opcional)
            language_expert: Experto lingüístico pre-entrenado (opcional)
            freeze_experts: Si es True, congela los pesos de los expertos
        """
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Componentes principales
        
        # Enrutador Neural Primario
        self.primary_router = PrimaryRouter(config)
        
        # Enrutadores específicos de dominio
        self.domain_routers = nn.ModuleDict({
            'arithmetic': ArithmeticRouter(config),
            'language': LanguageRouter(config)
        })
        
        # Expertos de dominio
        if math_expert is not None:
            self.experts = nn.ModuleDict({
                'arithmetic': math_expert,
                'language': language_expert if language_expert is not None else LanguageExpert(config)
            })
        else:
            self.experts = nn.ModuleDict({
                'arithmetic': MathExpert(config),
                'language': LanguageExpert(config)
            })
        
        # Módulo de fusión para combinar outputs de expertos
        self.fusion = ExpertFusion(config)
        
        # Sistema de memoria de trabajo
        if getattr(config, 'use_memory', True):
            self.memory = WorkingMemory(
                hidden_dim=config.hidden_dim,
                memory_size=getattr(config, 'memory_size', 128)
            )
            
            # Gestor de contexto (opcional)
            if getattr(config, 'use_context_manager', True):
                self.context_manager = ContextManager(
                    hidden_dim=config.hidden_dim,
                    memory_size=getattr(config, 'memory_size', 128)
                )
            else:
                self.context_manager = None
        else:
            self.memory = None
            self.context_manager = None
        
        # Congelar expertos si se solicita
        if freeze_experts:
            self.freeze_experts()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Procesa una entrada a través de VELORA.
        
        Args:
            hidden_states: Representación de entrada 
                - Para texto: [batch_size, seq_len, hidden_dim]
                - Para números: [batch_size, hidden_dim]
            attention_mask: Máscara de atención para secuencias [batch_size, seq_len]
            
        Returns:
            Tensor de salida y metadatos del procesamiento
        """
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Expandir dimensión si es necesario
        is_sequence = hidden_states.dim() == 3
        if not is_sequence:
            hidden_states = hidden_states.unsqueeze(1)
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1)
        
        # 1. Actualizar memoria de trabajo con entrada actual
        if self.memory is not None:
            memory_output, memory_metadata = self.memory(
                hidden_states, 
                mode="read"
            )
            
            # Enriquecer representación con contexto de memoria
            memory_enhanced = hidden_states + 0.2 * memory_output
        else:
            memory_enhanced = hidden_states
            memory_metadata = {}
        
        # 2. Enrutar la entrada a través del enrutador primario
        primary_routing = self.primary_router(memory_enhanced, attention_mask)
        
        # Mapeo de índices de dominio a nombres
        domain_map = {0: 'arithmetic', 1: 'language'}
        
        # 3. Procesamiento específico por dominio
        expert_outputs = []
        expert_routing_infos = []
        expert_metadata = []
        
        for i in range(batch_size):
            # Determinar dominio primario
            domain_idx = primary_routing['primary_domain'][i].item()
            domain_name = domain_map[domain_idx]
            
            # Extraer representación para este ejemplo
            if is_sequence:
                sample_input = hidden_states[i:i+1]
                sample_mask = None if attention_mask is None else attention_mask[i:i+1]
            else:
                sample_input = hidden_states[i]
                sample_mask = None
            
            # Obtener enrutamiento específico de dominio
            domain_router = self.domain_routers[domain_name]
            
            if domain_name == 'arithmetic':
                domain_routing = domain_router(
                    sample_input,
                    sample_mask,
                    None  # token_ids no disponibles en esta interfaz
                )
                
                # Obtener hint de operación
                operation_hint = domain_routing['operation'][0].item()
                operands = domain_routing['operands']
                
                # Activar experto aritmético
                expert = self.experts['arithmetic']
                output, metadata = expert(sample_input, operation_hint, operands, sample_mask)
            
            else:  # 'language'
                domain_routing = domain_router(sample_input, sample_mask)
                
                # Obtener hint de tarea lingüística
                task_hint = domain_routing['task'][0].item()
                
                # Activar experto lingüístico
                expert = self.experts['language']
                output, metadata = expert(sample_input, task_hint, sample_mask)
            
            # Guardar outputs y metadatos
            expert_outputs.append(output)
            expert_routing_infos.append(domain_routing)
            expert_metadata.append(metadata)
        
        # 4. Fusionar outputs de expertos
        expert_weights = primary_routing['expert_weights']
        fused_output, fusion_metadata = self.fusion(expert_outputs, expert_weights)
        
        # 5. Actualizar memoria de trabajo con resultado procesado
        if self.memory is not None:
            _, write_metadata = self.memory(
                fused_output,
                mode="write"
            )
            
            memory_metadata.update(write_metadata)
        
        # 6. Procesar a través del gestor de contexto si está disponible
        if self.context_manager is not None:
            context_output, context_metadata = self.context_manager(fused_output)
            
            # Combinar salida con información contextual
            final_output = (fused_output + context_output) / 2.0
            
            memory_metadata.update(context_metadata)
        else:
            final_output = fused_output
        
        # Compilar todos los metadatos
        metadata = {
            'primary_routing': primary_routing,
            'expert_routing': expert_routing_infos,
            'expert_metadata': expert_metadata,
            'fusion': fusion_metadata,
            'memory': memory_metadata
        }
        
        return final_output, metadata
    
    def explain(self, metadata: Dict[str, Any]) -> List[str]:
        """
        Genera explicaciones del procesamiento interno.
        
        Args:
            metadata: Metadatos del forward pass
            
        Returns:
            Lista de explicaciones en formato legible
        """
        explanations = []
        
        # Obtener explicación del enrutador primario
        primary_explanations = self.primary_router.explain_decisions(metadata['primary_routing'])
        
        # Combinar con explicaciones de expertos
        for i, primary_exp in enumerate(primary_explanations):
            # Obtener dominio
            domain_idx = metadata['primary_routing']['primary_domain'][i].item()
            domain_name = 'arithmetic' if domain_idx == 0 else 'language'
            
            # Obtener explicación específica del experto
            if hasattr(self.experts[domain_name], 'explain'):
                expert_exp = self.experts[domain_name].explain([metadata['expert_metadata'][i]])[0]
            else:
                expert_exp = f"Procesado por experto de {domain_name}"
            
            # Fusionar explicaciones
            full_exp = f"{primary_exp}\n\n{expert_exp}"
            
            # Añadir info sobre fusión
            if 'consistency_score' in metadata['fusion']:
                conf = metadata['fusion']['consistency_score'][i].item()
                full_exp += f"\n\nConfianza en resultado: {conf:.2f}"
            
            explanations.append(full_exp)
        
        return explanations
    
    def freeze_experts(self):
        """Congela los pesos de los expertos."""
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = False
    
    def unfreeze_experts(self):
        """Descongela los pesos de los expertos."""
        for expert in self.experts.values():
            for param in expert.parameters():
                param.requires_grad = True
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        Descongela capas específicas del modelo.
        
        Args:
            layer_names: Lista de nombres de capas a descongelar
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True