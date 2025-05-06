# src/models/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertFusion(nn.Module):
    """
    Módulo de fusión que integra las salidas de diferentes expertos.
    
    Ofrece:
    1. Fusión adaptativa basada en confianza
    2. Resolución de conflictos
    3. Verificación de consistencia
    """
    
    def __init__(self, input_dim=128, hidden_dim=64, num_experts=2):
        super(ExpertFusion, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Codificador para cada salida de experto
        self.expert_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Mecanismo de atención para fusión inteligente
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Red de fusión para combinar representaciones
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, input_dim)
        )
        
        # Verificador de consistencia
        self.consistency_checker = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, expert_outputs, expert_weights):
        """
        Fusiona salidas de expertos en una representación unificada.
        
        Args:
            expert_outputs: Lista de tensores de salida de expertos
            expert_weights: Pesos para cada experto
            
        Returns:
            Representación fusionada y metadatos
        """
        device = expert_weights.device
        batch_size = expert_weights.size(0)
        
        # Normalizar pesos de expertos
        normalized_weights = F.normalize(expert_weights, p=1, dim=1)
        
        # Procesar cada salida de experto
        processed_outputs = []
        for output in expert_outputs:
            # Asegurar dimensión de batch
            if output.dim() == 1:
                output = output.unsqueeze(0)
                
            # Codificar salida
            processed = self.expert_encoder(output)
            processed_outputs.append(processed)
        
        # Si hay más de una salida, aplicar fusión con atención
        if len(processed_outputs) > 1:
            # Apilar salidas para atención
            stacked = torch.stack(processed_outputs, dim=1)
            
            # Crear consulta utilizando el primer experto como base
            query = processed_outputs[0].unsqueeze(1)
            
            # Aplicar atención para fusionar expertos
            attn_output, attn_weights = self.fusion_attention(query, stacked, stacked)
            
            # Preparar para red de fusión
            fusion_input = attn_output.squeeze(1)
        else:
            # Si solo hay un experto, usar su salida directamente
            fusion_input = processed_outputs[0]
        
        # Aplicar red de fusión
        fused_output = self.fusion_network(fusion_input)
        
        # Verificar consistencia si hay múltiples expertos
        if len(expert_outputs) > 1:
            # Concatenar salidas para verificación
            concat_outputs = torch.cat([expert_outputs[0], expert_outputs[1]], dim=1)
            consistency_score = self.consistency_checker(concat_outputs)
        else:
            # Si solo hay un experto, asignar alta consistencia
            consistency_score = torch.ones((batch_size, 1), device=device) * 0.95
        
        return fused_output, {
            'consistency_score': consistency_score,
            'expert_influence': normalized_weights
        }
    
    def resolve_conflicts(self, expert_outputs, confidences):
        """
        Resuelve conflictos entre expertos usando confianza.
        
        Args:
            expert_outputs: Lista de salidas en conflicto
            confidences: Confianza de cada salida
            
        Returns:
            Representación resuelta
        """
        # Encontrar el experto más confiable
        max_confidence, max_idx = torch.max(confidences, dim=0)
        most_confident_output = expert_outputs[max_idx]
        
        # Si la confianza máxima es baja, aplicar fusión ponderada
        if max_confidence < 0.7:
            # Normalizar confianzas
            weights = F.normalize(confidences, p=1, dim=0)
            
            # Fusionar ponderadamente
            weighted_sum = torch.zeros_like(expert_outputs[0])
            for i, output in enumerate(expert_outputs):
                weighted_sum += output * weights[i]
                
            return weighted_sum
        else:
            # Si hay confianza alta, usar el experto más confiable
            return most_confident_output