"""
Experto de resta para VELORA.

Este módulo implementa el experto especializado en operaciones de resta.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SubtractExpert(nn.Module):
    """
    Experto de resta para el sistema VELORA.
    
    Implementa redes especializadas para operaciones de resta,
    con atención especial al manejo de orden de operandos y
    precisión con números negativos.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Parámetros de arquitectura
        self.hidden_dim = config.hidden_dim
        
        # Red neuronal para representación de resta
        self.subtract_network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Capa de proyección del resultado a espacio latente
        self.result_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.hidden_dim)
        )
        
        # Red para estimación de precisión/confianza
        self.precision_estimator = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Red para verificación de signo
        self.sign_verifier = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Tanh()  # Salida entre -1 y 1
        )
        
        # Inicialización específica
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicialización especial para optimizar operaciones de resta.
        Introduce inducción para operaciones simples.
        """
        # Inicializar última capa para aproximar operación de resta
        last_layer = self.subtract_network[-1]
        nn.init.tensor_([
            [1.0, -1.0]  # Primer operando positivo, segundo negativo
        ], last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        
        # Inicializaciones para otras capas
        for module in self.subtract_network[:-1]:
            if isinstance(module, nn.Linear):
                # Inicialización que favorece la preservación de signo
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        operands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza la operación de resta.
        
        Args:
            operands: Tensor con operandos [batch_size, 2]
            
        Returns:
            Resultado neural y resultado simbólico
        """
        batch_size = operands.size(0)
        
        # Extraer operandos
        a = operands[:, 0].unsqueeze(1)  # [batch_size, 1]
        b = operands[:, 1].unsqueeze(1)  # [batch_size, 1]
        
        # Calcular resultado mediante red neural
        neural_result = self.subtract_network(operands)
        
        # Calcular resultado simbólico (exacto)
        symbolic_result = a - b
        
        # Verificar signo para corregir errores potenciales
        predicted_sign = self.sign_verifier(operands)
        actual_sign = torch.sign(symbolic_result)
        
        # Si hay discrepancia de signo entre predicción y cálculo simbólico,
        # ajustar el resultado neural
        sign_match = (torch.sign(predicted_sign) == actual_sign).float()
        sign_corrected_result = sign_match * neural_result + (1 - sign_match) * symbolic_result
        
        # Estimar precisión/confianza
        precision = self.precision_estimator(operands)
        
        # Aplicar corrección basada en precisión
        # Si la precisión es alta, confiar más en resultado simbólico
        corrected_result = (1 - precision) * sign_corrected_result + precision * symbolic_result
        
        return corrected_result.squeeze(1), symbolic_result.squeeze(1)
    
    def encode_result(self, result: torch.Tensor) -> torch.Tensor:
        """
        Codifica el resultado numérico en espacio de representación.
        
        Args:
            result: Tensor con resultado numérico [batch_size]
            
        Returns:
            Representación codificada [batch_size, hidden_dim]
        """
        # Asegurar formato correcto
        if result.dim() == 1:
            result = result.unsqueeze(1)  # [batch_size, 1]
        
        # Codificar mediante red
        encoded = self.result_encoder(result)
        
        return encoded