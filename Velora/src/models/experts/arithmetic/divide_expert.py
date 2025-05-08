"""
Experto de división para VELORA.

Este módulo implementa el experto especializado en operaciones de división.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class DivideExpert(nn.Module):
    """
    Experto de división para el sistema VELORA.
    
    Implementa redes especializadas para operaciones de división,
    con atención especial a detección de división por cero y
    estabilidad numérica.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Parámetros de arquitectura
        self.hidden_dim = config.hidden_dim
        
        # Red neuronal para representación de división
        # Más profunda que suma/resta por la mayor complejidad
        self.divide_network = nn.Sequential(
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
        
        # Red logarítmica para división más estable
        self.log_divide_network = nn.Sequential(
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
        
        # Detector de división por cero
        self.zero_division_detector = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Red para detección de casos especiales
        self.special_case_detector = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Detecta: división por cero, división por uno, casos normales
            nn.Softmax(dim=1)
        )
        
        # Inicialización específica
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Inicialización especial para operaciones de división.
        """
        # Inicializaciones para red logarítmica
        for module in self.log_divide_network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Ajustar la última capa para aproximar resta (para implementar división via log)
        last_log_layer = self.log_divide_network[-1]
        nn.init.tensor_([
            [1.0, -1.0]  # Primer operando positivo, segundo negativo (log(a/b) = log(a) - log(b))
        ], last_log_layer.weight)
        nn.init.zeros_(last_log_layer.bias)
        
        # Inicializar detector de división por cero para ser muy sensible a valores cercanos a cero
        zero_detector_last = self.zero_division_detector[-2]
        nn.init.tensor_([
            [0.0, -10.0]  # Gran peso negativo para segundo operando
        ], zero_detector_last.weight)
        nn.init.ones_(zero_detector_last.bias)  # Bias alto para detectar incluso pequeños valores
    
    def forward(
        self, 
        operands: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Realiza la operación de división.
        
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
        
        # Detectar división por cero
        zero_div_score = self.zero_division_detector(operands)
        is_zero_division = (zero_div_score > 0.5) | (torch.abs(b) < 1e-6)
        
        # Detectar casos especiales
        special_cases = self.special_case_detector(operands)
        zero_case = special_cases[:, 0].unsqueeze(1)  # División por 0
        one_case = special_cases[:, 1].unsqueeze(1)   # División por 1
        
        # Calcular resultado mediante red neural estándar
        direct_result = self.divide_network(operands)
        
        # Calcular mediante aproximación logarítmica para mayor estabilidad numérica
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
        
        # Aplicar red (equivalente a restar logaritmos)
        log_output = self.log_divide_network(log_inputs)
        
        # Convertir de vuelta a espacio lineal y aplicar signo
        log_result = torch.exp(log_output) * result_sign
        
        # Calcular resultado simbólico (exacto)
        # Manejar división por cero
        safe_b = torch.where(
            torch.abs(b) < 1e-6,
            torch.ones_like(b) * 1e-6 * torch.sign(b + 1e-10),  # Usar signo preservado
            b
        )
        symbolic_result = a / safe_b
        
        # Combinar resultados de ambos métodos según magnitud
        magnitude_ratio = torch.abs(a) / (torch.abs(b) + epsilon)
        log_weight = torch.clamp(torch.log10(magnitude_ratio + 1) / 5.0, 0.0, 1.0)  # Peso para método log
        
        combined_result = (1 - log_weight) * direct_result + log_weight * log_result
        
        # Aplicar casos especiales detectados
        # Si divisor es 0, usar valor especial (NaN en versión real, aquí usamos 0 para simplificar)
        zero_div_result = torch.zeros_like(combined_result)
        # Si divisor es 1, resultado es dividendo
        one_div_result = a
        
        # Combinar según detección
        neural_result = torch.where(
            is_zero_division,
            zero_div_result,
            torch.where(
                torch.abs(b - 1.0) < 1e-6,
                one_div_result,
                combined_result
            )
        )
        
        # Para resultado simbólico, manejar explícitamente división por cero
        symbolic_result = torch.where(
            is_zero_division,
            torch.zeros_like(symbolic_result),  # O mejor, valor especial que indique error
            symbolic_result
        )
        
        # Estimar precisión/confianza
        base_precision = self.precision_estimator(operands)
        
        # Reducir confianza para divisiones cercanas a cero
        zero_penalty = torch.clamp(1.0 - 1000.0 * torch.abs(b), 0.0, 0.9).unsqueeze(1)
        precision = torch.clamp(base_precision - zero_penalty, 0.1, 1.0)
        
        # También reducir confianza para divisiones con cocientes muy grandes
        large_result_penalty = torch.clamp(
            torch.log10(torch.abs(symbolic_result) + 1.0) / 10.0, 
            0.0, 
            0.5
        ).unsqueeze(1)
        precision = torch.clamp(precision - large_result_penalty, 0.1, 1.0)
        
        # Aplicar corrección basada en precisión
        corrected_result = (1 - precision) * neural_result + precision * symbolic_result.unsqueeze(1)
        
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