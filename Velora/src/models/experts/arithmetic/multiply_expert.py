"""
Experto de multiplicación para VELORA.

Este módulo implementa el experto especializado en operaciones de multiplicación.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class MultiplyExpert(nn.Module):
    """
    Experto de multiplicación para el sistema VELORA.
    
    Implementa redes especializadas para operaciones de multiplicación,
    con atención especial a la escala de valores y preservación de precisión.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Parámetros de arquitectura
        self.hidden_dim = config.hidden_dim
        
        # Red neuronal para representación de multiplicación
        # Más profunda que suma/resta por la mayor complejidad
        self.multiply_network = nn.Sequential(
            nn.Linear(2, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Red logarítmica para multiplicación más estable
        self.log_multiply_network = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
        
        # Capa de proyección del resultado a espacio latente
        self.result_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, self.hidden_dim)
        )
        
        # Red para estimación de precisión/confianza
        self.precision_estimator = nn.Sequential(
            nn.Linear(2, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Red para detección de casos especiales
        self.special_case_detector = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Detecta: cero, uno, casos normales
            nn.Softmax(dim=1)
        )
        
        # Inicialización específica
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicialización especial para operaciones de multiplicación.
        """
        # Inicializaciones para red logarítmica
        for module in self.log_multiply_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Ajustar la última capa para aproximar suma (para implementar multiplicación via log)
        last_log_layer = self.log_multiply_network[-1]
        nn.init.ones_(last_log_layer.weight)
        nn.init.zeros_(last_log_layer.bias)
    
    def forward(
        self, 
        operands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza la operación de multiplicación.
        
        Args:
            operands: Tensor con operandos [batch_size, 2]
            
        Returns:
            Resultado neural y resultado simbólico
        """
        batch_size = operands.size(0)
        device = operands.device
        
        # Extraer operandos
        a = operands[:, 0].unsqueeze(1)  # [batch_size, 1]
        b = operands[:, 1].unsqueeze(1)  # [batch_size, 1]
        
        # Detectar casos especiales (multiplicación por 0 o 1)
        special_cases = self.special_case_detector(operands)
        zero_case = special_cases[:, 0].unsqueeze(1)  # Multiplicación por 0
        one_case = special_cases[:, 1].unsqueeze(1)   # Multiplicación por 1
        
        # Calcular resultado mediante red neural estándar
        direct_result = self.multiply_network(operands)
        
        # Calcular mediante aproximación logarítmica para mayor estabilidad numérica
        # con valores grandes
        
        # Añadir pequeño epsilon para evitar log(0)
        epsilon = 1e-7
        
        # Calcular signo del resultado
        sign_a = torch.sign(a)
        sign_b = torch.sign(b)
        result_sign = sign_a * sign_b
        
        # Trabajar con valores absolutos para logaritmos
        abs_a = torch.abs(a) + epsilon
        abs_b = torch.abs(b) + epsilon
        
        # Convertir a espacio logarítmico
        log_a = torch.log(abs_a)
        log_b = torch.log(abs_b)
        
        # Concatenar logaritmos
        log_inputs = torch.cat([log_a, log_b], dim=1)
        
        # Aplicar red (equivalente a sumar logaritmos)
        log_output = self.log_multiply_network(log_inputs)
        
        # Convertir de vuelta a espacio lineal y aplicar signo
        log_result = torch.exp(log_output) * result_sign
        
        # Calcular resultado simbólico (exacto)
        symbolic_result = a * b
        
        # Combinar resultados de ambos métodos según magnitud
        # Para valores pequeños, usar método directo
        # Para valores grandes, usar método logarítmico
        magnitude = torch.max(torch.abs(a), torch.abs(b))
        log_weight = torch.clamp(magnitude / 100.0, 0.0, 1.0)  # Peso para método log
        
        combined_result = (1 - log_weight) * direct_result + log_weight * log_result
        
        # Aplicar casos especiales detectados
        # Si algún operando es 0, resultado es 0
        zero_result = torch.zeros_like(combined_result)
        # Si algún operando es 1, resultado es el otro operando
        one_result = torch.where(
            torch.abs(a - 1.0) < torch.abs(b - 1.0),
            b,  # Si a ≈ 1, resultado es b
            a   # Si b ≈ 1, resultado es a
        )
        
        # Combinar según detección
        neural_result = (
            zero_case * zero_result +
            one_case * one_result +
            (1 - zero_case - one_case) * combined_result
        )
        
        # Estimar precisión/confianza
        precision = self.precision_estimator(operands)
        
        # Si la magnitud es muy grande, confiar más en resultado simbólico
        large_magnitude = (magnitude > 1000).float().unsqueeze(1)
        precision = torch.max(precision, large_magnitude)
        
        # Aplicar corrección basada en precisión
        corrected_result = (1 - precision) * neural_result + precision * symbolic_result
        
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