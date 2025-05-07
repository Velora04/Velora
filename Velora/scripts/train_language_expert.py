#!/usr/bin/env python
"""
Script para entrenar el experto de lenguaje de VELORA.

Este script entrena solamente el experto de lenguaje usando datos específicos
para tareas de procesamiento de lenguaje natural.
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# Asegurar que el módulo raíz esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experts.language_expert import LanguageExpert
from config.language_expert_config import LanguageExpertConfig
from csv_dataloader import VeloraCSVDataset

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento del experto de lenguaje de VELORA')
    
    parser.add_argument('--data_file', type=str, default='data/language_dataset.csv',
                      help='Archivo CSV con datos de entrenamiento de lenguaje')
    
    parser.add_argument('--val_file', type=str, default=None,
                      help='Archivo CSV con datos de validación (opcional)')
    
    parser.add_argument('--output_dir', type=str, default='models/language_expert',
                      help='Directorio donde guardar los modelos entrenados')
    
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Tamaño del batch para entrenamiento')
    
    parser.add_argument('--epochs', type=int, default=200,
                      help='Número de épocas de entrenamiento')
    
    parser.add_argument('--lr', type=float, default=2e-4,
                      help='Tasa de aprendizaje inicial')
    
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                      help='Factor de weight decay para el optimizador')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Tasa de dropout')
    
    parser.add_argument('--save_interval', type=int, default=10,
                      help='Guardar modelo cada N épocas')
    
    parser.add_argument('--warmup_steps', type=int, default=1000,
                      help='Pasos de calentamiento para el scheduler')
    
    parser.add_argument('--no_cuda', action='store_true',
                      help='Deshabilitar uso de CUDA incluso si está disponible')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Semilla para generación de números aleatorios')
    
    return parser.parse_args()

def create_language_dataloaders(data_file, val_file=None, batch_size=32, val_split=0.2, input_dim=128, seed=42):
    """
    Crea dataloader para entrenamiento y validación del experto de lenguaje.
    
    Args:
        data_file: Archivo CSV con datos de lenguaje
        val_file: Archivo CSV con datos de validación (opcional)
        batch_size: Tamaño del batch
        val_split: Proporción de datos para validación si no se proporciona val_file
        input_dim: Dimensión de entrada del modelo
        seed: Semilla aleatoria
        
    Returns:
        train_loader, val_loader
    """
    # Verificar si existe el archivo de datos
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"No se encontró el archivo de datos: {data_file}")
    
    # Cargar dataset completo
    full_dataset = VeloraCSVDataset(data_file, input_dim=input_dim)
    
    # Filtrar solo ejemplos de lenguaje (dominio=1)
    language_indices = [i for i in range(len(full_dataset)) if full_dataset.samples[i]['domain'] == 1]
    
    # Si no hay datos de lenguaje, mostrar error
    if len(language_indices) == 0:
        raise ValueError(f"No se encontraron ejemplos de lenguaje en {data_file}")
    
    print(f"Encontrados {len(language_indices)} ejemplos de lenguaje en {data_file}")
    
    # Función de colación para datos de lenguaje
    def language_collate_fn(batch):
        inputs, metadatas = zip(*batch)
        
        # Para experto de lenguaje, los inputs son secuencias
        # Calcular longitud máxima
        max_len = max(x.size(0) for x in inputs)
        batch_size = len(inputs)
        
        # Crear tensor con padding
        batch_inputs = torch.zeros(batch_size, max_len, input_dim)
        
        # Llenar tensor con datos
        for i, input_data in enumerate(inputs):
            seq_len = input_data.size(0)
            batch_inputs[i, :seq_len, :] = input_data
        
        # Procesar metadatas
        batch_metadata = {}
        keys = metadatas[0].keys()
        
        for key in keys:
            if isinstance(metadatas[0][key], torch.Tensor):
                batch_metadata[key] = torch.cat([m[key] for m in metadatas])
        
        return batch_inputs, batch_metadata
    
    # Si se proporciona archivo de validación específico
    if val_file and os.path.exists(val_file):
        val_dataset = VeloraCSVDataset(val_file, input_dim=input_dim)
        val_indices = [i for i in range(len(val_dataset)) if val_dataset.samples[i]['domain'] == 1]
        
        if len(val_indices) == 0:
            print(f"Advertencia: No se encontraron ejemplos de lenguaje en {val_file}")
            
            # Crear división de validación del conjunto principal
            torch.manual_seed(seed)
            train_size = int(len(language_indices) * (1 - val_split))
            train_indices = language_indices[:train_size]
            val_indices = language_indices[train_size:]
            
            # Crear subconjuntos
            train_subset = torch.utils.data.Subset(full_dataset, train_indices)
            val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        else:
            print(f"Encontrados {len(val_indices)} ejemplos de lenguaje en {val_file}")
            
            # Usar conjunto de entrenamiento completo y conjunto de validación específico
            train_subset = torch.utils.data.Subset(full_dataset, language_indices)
            val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    else:
        # Crear división de validación del conjunto principal
        torch.manual_seed(seed)
        train_size = int(len(language_indices) * (1 - val_split))
        train_indices = language_indices[:train_size]
        val_indices = language_indices[train_size:]
        
        # Crear subconjuntos
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=language_collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=language_collate_fn
    )
    
    print(f"DataLoader de entrenamiento: {len(train_subset)} ejemplos")
    print(f"DataLoader de validación: {len(val_subset)} ejemplos")
    
    return train_loader, val_loader

def train_language_expert():
    """Función principal de entrenamiento del experto de lenguaje."""
    # Parsear argumentos
    args = parse_args()
    
    # Establecer semilla para reproducibilidad
    torch.manual_seed(args.seed)
    
    # Configurar dispositivo
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Cargar configuración y crear modelo
    config = LanguageExpertConfig(
        input_dim=128,
        hidden_dim=256,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        warmup_steps=args.warmup_steps
    )
    
    model = LanguageExpert(config)
    model.to(device)
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Guardar configuración
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        # Convertir config a dict
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=4)
    
    # Crear dataloaders
    train_loader, val_loader = create_language_dataloaders(
        data_file=args.data_file,
        val_file=args.val_file,
        batch_size=args.batch_size,
        input_dim=config.input_dim,
        seed=args.seed
    )
    
    # Definir optimizador, scheduler y criterio
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Usar un scheduler con warmup para lenguaje
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Criterio compuesto para diferentes aspectos del experto de lenguaje
    # CrossEntropy para clasificación de tipo de consulta
    query_criterion = nn.CrossEntropyLoss()
    
    # CrossEntropy para predicción de tokens
    token_criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = padding
    
    # MSE para representación de resultado
    repr_criterion = nn.MSELoss()
    
    # Inicializar variables para seguimiento
    best_val_loss = float('inf')
    start_time = time.time()
    history = {
        'train_loss': [],
        'val_loss': [],
        'query_classification_accuracy': [],
        'learning_rate': []
    }
    
    # Ciclo de entrenamiento
    print("\nComenzando entrenamiento...")
    
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            
            # Modo entrenamiento
            model.train()
            train_loss = 0.0
            query_correct = 0
            total_samples = 0
            
            for batch_idx, (inputs, metadata) in enumerate(train_loader):
                inputs = inputs.to(device)
                target_query_type = metadata['language_task'].to(device)
                
                # Forward pass
                output, output_metadata = model(inputs)
                
                # Calcular pérdidas
                # Pérdida de clasificación de tipo de consulta
                query_loss = query_criterion(output_metadata['query_logits'], target_query_type)
                
                # Pérdida total (aquí podemos agregar más componentes si es necesario)
                loss = query_loss
                
                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Actualizar estadísticas
                train_loss += loss.item() * inputs.size(0)
                
                # Calcular precisión de clasificación de tipo de consulta
                pred_query_type = output_metadata['query_type']
                query_correct += (pred_query_type == target_query_type).sum().item()
                total_samples += inputs.size(0)
                
                # Mostrar progreso
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.6f}")
            
            # Actualizar scheduler
            scheduler.step()
            
            # Calcular pérdida y precisión de entrenamiento promedio
            train_loss /= total_samples
            query_accuracy = query_correct / total_samples
            
            # Modo evaluación
            model.eval()
            val_loss = 0.0
            val_query_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for inputs, metadata in val_loader:
                    inputs = inputs.to(device)
                    target_query_type = metadata['language_task'].to(device)
                    
                    # Forward pass
                    output, output_metadata = model(inputs)
                    
                    # Calcular pérdida de clasificación de tipo de consulta
                    query_loss = query_criterion(output_metadata['query_logits'], target_query_type)
                    
                    # Pérdida total
                    loss = query_loss
                    
                    # Actualizar estadísticas
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Calcular precisión de clasificación de tipo de consulta
                    pred_query_type = output_metadata['query_type']
                    val_query_correct += (pred_query_type == target_query_type).sum().item()
                    val_samples += inputs.size(0)
            
            # Calcular pérdida y precisión de validación promedio
            val_loss /= val_samples
            val_query_accuracy = val_query_correct / val_samples
            
            # Obtener tasa de aprendizaje actual
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar estadísticas en historial
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['query_classification_accuracy'].append(val_query_accuracy)
            history['learning_rate'].append(current_lr)
            
            # Guardar historial en cada época
            with open(os.path.join(args.output_dir, "history.json"), 'w') as f:
                json.dump(history, f, indent=4)
            
            # Calcular tiempo de época
            epoch_time = time.time() - epoch_start_time
            
            # Mostrar estadísticas de época
            print(f"Epoch {epoch}/{args.epochs} completada en {epoch_time:.2f}s")
            print(f"  Pérdida entrenamiento: {train_loss:.6f}")
            print(f"  Pérdida validación: {val_loss:.6f}")
            print(f"  Precisión clasificación (train): {query_accuracy:.4f}")
            print(f"  Precisión clasificación (val): {val_query_accuracy:.4f}")
            print(f"  Learning rate: {current_lr}")
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"  Guardado mejor modelo con val_loss={val_loss:.6f}")
            
            # Guardar checkpoint periódico
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'query_accuracy': val_query_accuracy
                }, checkpoint_path)
                print(f"  Guardado checkpoint en época {epoch}")
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time:.2f} segundos")
    print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
    
    # Guardar modelo final
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")
    
    return model, history

if __name__ == "__main__":
    train_language_expert()