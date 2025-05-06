# src/data/dataset.py
import torch
from torch.utils.data import Dataset

class VeloraDataset(Dataset):
    """
    Dataset para entrenar el modelo VELORA.
    
    Genera muestras sintéticas de:
    1. Operaciones matemáticas
    2. Secuencias de lenguaje natural
    """
    
    def __init__(self, num_samples=1000, input_dim=128, split='train'):
        """
        Inicializa el dataset.
        
        Args:
            num_samples: Número de muestras a generar
            input_dim: Dimensión de entrada
            split: 'train' o 'val'
        """
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.split = split
        
        # Generar datos
        self.data = []
        self.targets = []
        
        # Peso relativo de cada dominio
        math_ratio = 0.5
        
        # Generar muestras
        for i in range(num_samples):
            # Alternar entre matemáticas y lenguaje
            if i < num_samples * math_ratio:
                # Crear muestra matemática
                x, target = self._generate_math_sample()
            else:
                # Crear muestra de lenguaje
                x, target = self._generate_language_sample()
            
            self.data.append(x)
            self.targets.append(target)
    
    # src/data/dataset.py - Corrección para el método _generate_math_sample

    def _generate_math_sample(self):
        """Genera una muestra de operación matemática con dimensiones consistentes."""
        # Generar números aleatorios
        a = torch.rand(1) * 10
        b = torch.rand(1) * 10

        # Seleccionar operación aleatoria
        operation = torch.randint(0, 4, (1,)).item()  # 0=suma, 1=resta, 2=mult, 3=div

        # Calcular resultado
        if operation == 0:  # suma
            result = a + b
        elif operation == 1:  # resta
            result = a - b
        elif operation == 2:  # multiplicación
            result = a * b
        else:  # división
            # Evitar división por cero
            if b.item() < 0.1:
                b = torch.tensor([0.1])
            result = a / b

        # Crear entrada con forma consistente [1, input_dim]
        x = torch.zeros(1, self.input_dim)
        x[0, 0] = a
        x[0, 1] = b

        # Crear target
        target = {
            'domain': torch.tensor([0]),  # 0 = matemáticas
            'math_operation': torch.tensor([operation]),
            'language_task': torch.tensor([0]),  # valor dummy
            'result': torch.zeros(1, self.input_dim)  # resultado con dimensión consistente
        }

        # Almacenar el resultado en la posición adecuada
        target['result'][0, 0] = result

        return x, target

    def _generate_language_sample(self):
        """Genera una muestra de lenguaje con dimensiones consistentes."""
        # Determinar longitud de secuencia
        seq_len = torch.randint(5, 20, (1,)).item()
    
        # Crear secuencia con forma consistente [seq_len, input_dim]
        x = torch.randn(seq_len, self.input_dim)
    
        # Seleccionar tipo de tarea aleatoria
        task = torch.randint(0, 3, (1,)).item()  # 0=pregunta, 1=comando, 2=declaración
    
        # Crear target con forma consistente
        target = {
            'domain': torch.tensor([1]),  # 1 = lenguaje
            'math_operation': torch.tensor([0]),  # valor dummy
            'language_task': torch.tensor([task]),
            'result': torch.randn(1, self.input_dim)  # misma forma que para matemáticas
        }

        return x, target
    
    def __len__(self):
        """Retorna el número de muestras."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """Retorna una muestra por índice."""
        return self.data[idx], self.targets[idx]