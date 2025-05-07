"""
Cargador de datos CSV para VELORA.

Este módulo implementa las clases y funciones necesarias para cargar y procesar
datos en formato CSV para el entrenamiento de los modelos VELORA.
"""
import os
import csv
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class VeloraCSVDataset(Dataset):
    """
    Dataset para cargar datos CSV generados para VELORA.
    
    Formato esperado del CSV:
    - text: El texto de entrada
    - domain: El dominio (0 para matemáticas, 1 para lenguaje)
    - operation: La operación matemática
    - operand1 y operand2: Los operandos para matemáticas
    - expected_result: El resultado esperado
    - query_type: El tipo de consulta de lenguaje
    """
    
    def __init__(self, csv_file, input_dim=128, text_embedding_dim=64):
        """
        Inicializa el dataset.
        
        Args:
            csv_file: Ruta al archivo CSV con los datos
            input_dim: Dimensión de los vectores de entrada
            text_embedding_dim: Dimensión para codificar textos
        """
        self.input_dim = input_dim
        self.text_embedding_dim = text_embedding_dim
        
        # Cargar datos del CSV
        self.samples = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convertir tipos
                sample = {
                    'text': row['text'],
                    'domain': int(row['domain']),
                }
                
                # Procesar campos específicos del dominio
                if sample['domain'] == 0:  # Matemáticas
                    sample['operation'] = int(row['operation'])
                    sample['operand1'] = float(row['operand1'])
                    sample['operand2'] = float(row['operand2'])
                    sample['expected_result'] = float(row['expected_result'])
                else:  # Lenguaje
                    sample['query_type'] = int(row['query_type']) if row['query_type'] else 0
                
                self.samples.append(sample)
    
    def __len__(self):
        """Retorna el número de muestras."""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Retorna una muestra procesada.
        
        Para muestras matemáticas: tensor de 2 elementos [operand1, operand2]
        Para muestras de lenguaje: secuencia de vectores simulando embeddings
        """
        sample = self.samples[idx]
        
        if sample['domain'] == 0:  # Matemáticas
            # Crear tensor con los dos operandos
            x = torch.zeros(self.input_dim)
            x[0] = sample['operand1']
            x[1] = sample['operand2']
            
            # Crear metadata
            metadata = {
                'domain': torch.tensor([sample['domain']]),
                'math_operation': torch.tensor([sample['operation']]),
                'language_task': torch.tensor([0]),  # Dummy
                'result': torch.tensor([sample['expected_result']])
            }
            
        else:  # Lenguaje
            # Simular embeddings para el texto
            # En un caso real usaríamos un tokenizer y embeddings
            text_len = min(len(sample['text']), 20)  # Limitar longitud
            
            # Crear secuencia de vectores aleatorios para simular embeddings
            # Use deterministic "embeddings" based on character values
            x = torch.zeros(text_len, self.input_dim)
            for i, char in enumerate(sample['text'][:text_len]):
                # Use character code as a seed for vector
                seed = ord(char)
                np.random.seed(seed)
                x[i, :self.text_embedding_dim] = torch.tensor(
                    np.random.normal(0, 1, self.text_embedding_dim)
                )
            
            # Crear metadata
            metadata = {
                'domain': torch.tensor([sample['domain']]),
                'math_operation': torch.tensor([0]),  # Dummy
                'language_task': torch.tensor([sample['query_type']]),
                'result': torch.tensor([0.0])  # Dummy result con dimensión consistente
            }
        
        return x, metadata

def create_dataloaders_from_csv(train_file, val_file, batch_size=32, input_dim=128):
    """
    Crea dataloaders a partir de archivos CSV.
    
    Args:
        train_file: Ruta al archivo CSV de entrenamiento
        val_file: Ruta al archivo CSV de validación
        batch_size: Tamaño del batch
        input_dim: Dimensión de entrada
        
    Returns:
        train_loader, val_loader
    """
    # Verificar que los archivos existen
    for file_path in [train_file, val_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
    
    # Crear datasets
    train_dataset = VeloraCSVDataset(train_file, input_dim=input_dim)
    val_dataset = VeloraCSVDataset(val_file, input_dim=input_dim)
    
    print(f"Dataset de entrenamiento: {len(train_dataset)} muestras")
    print(f"Dataset de validación: {len(val_dataset)} muestras")
    
    # Función de colación para manejar datos de longitud variable
    def collate_fn(batch):
        inputs, metadatas = zip(*batch)
        
        # Verificar si son secuenciales o no
        is_sequential = any(x.dim() > 1 for x in inputs)
        
        if is_sequential:
            # Para datos secuenciales (lenguaje)
            max_len = max(x.size(0) for x in inputs)
            batch_size = len(inputs)
            
            # Crear tensor con padding
            batch_inputs = torch.zeros(batch_size, max_len, input_dim)
            
            for i, input_data in enumerate(inputs):
                seq_len = input_data.size(0)
                batch_inputs[i, :seq_len, :] = input_data
        else:
            # Para datos no secuenciales (matemáticas)
            batch_inputs = torch.stack(inputs)
        
        # Procesar metadatas asegurando compatibilidad de dimensiones
        batch_metadata = {}
        keys = metadatas[0].keys()
        
        for key in keys:
            if isinstance(metadatas[0][key], torch.Tensor):
                # Comprobar dimensiones antes de concatenar
                meta_tensors = [m[key] for m in metadatas]
                
                # Si el key es 'result', verificar y ajustar dimensiones
                if key == 'result':
                    # Asegurar que todos los tensores tienen la misma forma
                    shapes = [t.shape for t in meta_tensors]
                    if not all(s == shapes[0] for s in shapes):
                        # Ajustar tensores a una forma consistente
                        adjusted_tensors = []
                        for t in meta_tensors:
                            if t.dim() == 0:  # Escalar
                                t = t.unsqueeze(0)
                            elif t.dim() > 1:  # Tensor multidimensional
                                t = t.view(-1)
                            adjusted_tensors.append(t)
                        meta_tensors = adjusted_tensors
                
                # Concatenar tensores
                try:
                    batch_metadata[key] = torch.cat(meta_tensors)
                except RuntimeError as e:
                    # Si hay error, imprimir información detallada para depurar
                    print(f"Error concatenando tensores para key '{key}'")
                    print(f"Formas de los tensores: {[t.shape for t in meta_tensors]}")
                    raise e
        
        return batch_inputs, batch_metadata
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader