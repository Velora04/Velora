# src/models/velora.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.router import NeuralRouter
from src.models.experts.math_expert import MathExpert
from src.models.experts.language_expert import LanguageExpert
from src.models.fusion import ExpertFusion

class VELORA(nn.Module):
    """
    VELORA: Versatile Language and Operations Reasoning Architecture
    
    Un modelo de IA modular que combina expertos especializados para
    diferentes dominios (matemáticas y lenguaje natural) mediante un
    sistema de enrutamiento neuronal.
    """
    
    def __init__(self, config):
        """
        Inicializa el modelo VELORA.
        
        Args:
            config: Configuración del modelo (dimensiones, capas, etc.)
        """
        super(VELORA, self).__init__()
        
        # Configuración
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.latent_dim = config.latent_dim
        
        # Componentes principales
        self.router = NeuralRouter(
            input_dim=config.input_dim,
            hidden_dim=config.router_hidden_dim,
            num_experts=config.num_experts
        )
        
        # Expertos
        self.experts = nn.ModuleDict({
            'math': MathExpert(
                input_dim=config.input_dim,
                hidden_dim=config.expert_hidden_dim
            ),
            'language': LanguageExpert(
                input_dim=config.input_dim,
                hidden_dim=config.expert_hidden_dim,
                vocab_size=config.vocab_size
            )
        })
        
        # Módulo de fusión
        self.fusion = ExpertFusion(
            input_dim=config.input_dim,
            hidden_dim=config.fusion_hidden_dim
        )
        
        # Memoria de trabajo (opcional para futuras expansiones)
        self.working_memory = nn.Parameter(
            torch.zeros(1, config.memory_size, config.input_dim)
        )
        
    def forward(self, x):
        """
        Procesa una entrada a través de VELORA.
        
        Args:
            x: Tensor de entrada
                - Para texto: [batch_size, seq_len, input_dim]
                - Para números: [batch_size, input_dim]
        
        Returns:
            Tensor de salida y metadatos del procesamiento
        """
        # 1. Enrutar la entrada a los expertos adecuados
        routing_info = self.router(x)
        
        # 2. Activar expertos según decisiones de enrutamiento
        expert_outputs = []
        expert_metadata = []
        
        batch_size = x.size(0)
        
        # Mapeo de índices a nombres de expertos
        experts_map = {0: 'math', 1: 'language'}
        
        # Procesar cada ejemplo en el batch
        for i in range(batch_size):
            # Determinar el experto principal
            domain_idx = routing_info['primary_domain'][i].item()
            expert_name = experts_map[domain_idx]
            
            # Preparar entrada para el experto (garantizar dimensiones correctas)
            if x.dim() == 3:  # [batch, seq_len, dim]
                expert_input = x[i:i+1]
            else:  # [batch, dim]
                expert_input = x[i:i+1]
            
            # Activar experto correspondiente
            if expert_name == 'math':
                # Pasar información de operación matemática
                math_op = routing_info['math_operation'][i].item()
                output, metadata = self.experts['math'](
                    expert_input, 
                    operation_hint=math_op
                )
            else:  # 'language'
                # Pasar información de tarea lingüística
                lang_task = routing_info['language_task'][i].item()
                output, metadata = self.experts['language'](
                    expert_input,
                    task_hint=lang_task
                )
            
            # Almacenar salidas y metadatos
            expert_outputs.append(output)
            expert_metadata.append(metadata)
        
        # 3. Fusionar salidas de expertos
        expert_weights = routing_info['expert_weights']
        final_output, fusion_metadata = self.fusion(expert_outputs, expert_weights)
        
        # 4. Actualizar memoria de trabajo (opcional)
        if self.config.use_memory:
            # Mecanismo simple de actualización de memoria
            memory_gate = torch.sigmoid(self.memory_gate(final_output))
            self.working_memory = (
                memory_gate * self.working_memory + 
                (1 - memory_gate) * final_output.unsqueeze(1)
            )
        
        return final_output, {
            'routing': routing_info,
            'expert_metadata': expert_metadata,
            'fusion': fusion_metadata
        }
    
    def explain(self, metadata):
        """
        Genera explicaciones del procesamiento interno.
        
        Args:
            metadata: Metadatos del forward pass
        
        Returns:
            Explicaciones en formato legible
        """
        explanations = []
        
        # Obtener explicación del enrutador
        routing_explanations = self.router.explain_routing(metadata['routing'])
        
        # Combinar con explicaciones de expertos
        for i, route_exp in enumerate(routing_explanations):
            # Determinar qué experto procesó la entrada
            domain_idx = metadata['routing']['primary_domain'][i].item()
            expert_name = 'math' if domain_idx == 0 else 'language'
            
            # Obtener explicación específica del experto
            if hasattr(self.experts[expert_name], 'explain'):
                expert_exp = self.experts[expert_name].explain(metadata['expert_metadata'][i])
            else:
                expert_exp = f"Procesado por experto de {expert_name}"
            
            # Fusionar explicaciones
            full_exp = f"{route_exp}\n{expert_exp}"
            
            # Añadir info sobre fusión
            if 'consistency_score' in metadata['fusion']:
                conf = metadata['fusion']['consistency_score'][i].item()
                full_exp += f"\nConfianza en resultado: {conf:.2f}"
            
            explanations.append(full_exp)
        
        return explanations