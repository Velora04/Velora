# src/models/experts/math_expert.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MathExpert(nn.Module):
    """
    Experto en matemáticas basado en redes neuronales.
    
    Capaz de:
    1. Reconocer operaciones matemáticas
    2. Extraer números de la representación
    3. Realizar cálculos precisos
    4. Representar resultados en el espacio latente
    """
    
    def __init__(self, input_dim=128, hidden_dim=256):
        super(MathExpert, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Codificador para procesar entradas
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Detector de operación
        self.operation_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)  # 4 operaciones básicas
        )
        
        # Extractor de números
        self.number_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # Extrae 2 números
        )
        
        # Redes específicas para cada operación
        self.operation_networks = nn.ModuleDict({
            'add': nn.Sequential(
                nn.Linear(2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ),
            'subtract': nn.Sequential(
                nn.Linear(2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ),
            'multiply': nn.Sequential(
                nn.Linear(2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            ),
            'divide': nn.Sequential(
                nn.Linear(2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        })
        
        # Codificador de resultados
        self.result_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        
    def forward(self, x, operation_hint=None):
        """
        Procesa entrada matemática y calcula resultado.
        
        Args:
            x: Tensor de entrada
            operation_hint: Índice opcional de operación a realizar
            
        Returns:
            Representación del resultado y metadatos
        """
        # Manejar diferentes formatos de entrada
        if x.dim() > 2:
            # Si es una secuencia, aplicar pooling
            x = torch.mean(x, dim=1)
        
        # Codificar entrada
        hidden = self.encoder(x)
        
        # Detectar operación (si no se proporciona)
        if operation_hint is None:
            op_logits = self.operation_detector(hidden)
            operation = torch.argmax(op_logits, dim=1)
        else:
            operation = operation_hint if isinstance(operation_hint, torch.Tensor) else torch.tensor([operation_hint], device=x.device)
        
        # Extraer números
        numbers = self.number_extractor(hidden)
        
        # Realizar operación usando redes especializadas
        batch_size = x.size(0)
        results = torch.zeros(batch_size, 1, device=x.device)
        
        # Mapeo de operaciones
        op_mapping = {0: 'add', 1: 'subtract', 2: 'multiply', 3: 'divide'}
        
        for i in range(batch_size):
            op_idx = operation[i].item()
            op_name = op_mapping[op_idx]
            
            # Extraer par de números
            num_pair = numbers[i].unsqueeze(0)
            
            # Aplicar red específica para la operación
            results[i] = self.operation_networks[op_name](num_pair)
            
            # Verificar validez para división
            if op_idx == 3 and numbers[i, 1].item() == 0:
                # En caso de división por cero, establecer un resultado especial
                results[i] = torch.tensor([float('nan')], device=x.device)
        
        # Codificar resultado en formato de salida
        result_repr = self.result_encoder(results)
        
        # Añadir operación simbólica directa para validación
        direct_results = torch.zeros_like(results)
        
        for i in range(batch_size):
            op_idx = operation[i].item()
            a, b = numbers[i, 0].item(), numbers[i, 1].item()
            
            if op_idx == 0:  # suma
                direct_results[i] = a + b
            elif op_idx == 1:  # resta
                direct_results[i] = a - b
            elif op_idx == 2:  # multiplicación
                direct_results[i] = a * b
            else:  # división
                direct_results[i] = a / b if b != 0 else float('nan')
        
        return result_repr, {
            'operation': operation,
            'numbers': numbers,
            'neural_result': results.squeeze(-1),
            'symbolic_result': direct_results.squeeze(-1)
        }
    
    def explain(self, metadata):
        """
        Genera explicación de la operación realizada.
        
        Args:
            metadata: Metadatos del procesamiento
            
        Returns:
            Explicación en formato de texto
        """
        op_names = ['suma', 'resta', 'multiplicación', 'división']
        
        explanations = []
        batch_size = metadata['operation'].size(0)
        
        for i in range(batch_size):
            op_idx = metadata['operation'][i].item() 
            
            # Extraer números y resultados
            a, b = metadata['numbers'][i, 0].item(), metadata['numbers'][i, 1].item()
            neural_result = metadata['neural_result'][i].item()
            symbolic_result = metadata['symbolic_result'][i].item()
            
            # Crear explicación detallada
            explanation = f"Operación: {op_names[op_idx]}\n"
            explanation += f"Números: {a:.2f} y {b:.2f}\n"
            explanation += f"Resultado (red neuronal): {neural_result:.2f}\n"
            explanation += f"Resultado (simbólico): {symbolic_result:.2f}"
            
            explanations.append(explanation)
        
        return explanations