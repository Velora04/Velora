#!/usr/bin/env python
"""
Script para entrenar el experto matemático de VELORA.

Este script entrena solamente el experto matemático usando datos específicos
para operaciones matemáticas.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# Asegurar que el módulo raíz esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experts.math_expert import MathExpert
from config.math_expert_config import MathExpertConfig
from csv_dataloader import VeloraCSVDataset

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento del experto matemático de VELORA')
    
    parser.add_argument('--data_file', type=str, default='data/math_dataset.csv',
                      help='Archivo CSV con datos de entrenamiento matemático')
    
    parser.add_argument('--val_file', type=str, default=None,
                      help='Archivo CSV con datos de validación (opcional)')
    
    parser.add_argument('--output_dir', type=str, default='models/math_expert',
                      help='Directorio donde guardar los modelos entrenados')
    
    parser.add_argument('--batch_size', type=int, default=64,
                      help='Tamaño del batch para entrenamiento')
    
    parser.add_argument('--epochs', type=int, default=150,
                      help='Número de épocas de entrenamiento')
    
    parser.add_argument('--lr', type=float, default=3e-4,
                      help='Tasa de aprendizaje inicial')
    
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                      help='Factor de weight decay para el optimizador')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Tasa de dropout')
    
    parser.add_argument('--save_interval', type=int, default=10,
                      help='Guardar modelo cada N épocas')
    
    parser.add_argument('--no_cuda', action='store_true',
                      help='Deshabilitar uso de CUDA incluso si está disponible')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Semilla para generación de números aleatorios')
    
    return parser.parse_args()

def create_math_dataloaders(data_file, val_file=None, batch_size=64, val_split=0.2, input_dim=128, seed=42):
    """
    Crea dataloader para entrenamiento y validación del experto matemático.
    
    Args:
        data_file: Archivo CSV con datos matemáticos
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
    
    # Filtrar solo ejemplos matemáticos (dominio=0)
    math_indices = [i for i in range(len(full_dataset)) if full_dataset.samples[i]['domain'] == 0]
    
    # Si no hay datos matemáticos, mostrar error
    if len(math_indices) == 0:
        raise ValueError(f"No se encontraron ejemplos matemáticos en {data_file}")
    
    print(f"Encontrados {len(math_indices)} ejemplos matemáticos en {data_file}")
    
    # Función de colación para datos matemáticos
    def math_collate_fn(batch):
        inputs, metadatas = zip(*batch)
        
        # Para experto matemático, los inputs son tensores planos
        batch_inputs = torch.stack(inputs)
        
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
        val_indices = [i for i in range(len(val_dataset)) if val_dataset.samples[i]['domain'] == 0]
        
        if len(val_indices) == 0:
            print(f"Advertencia: No se encontraron ejemplos matemáticos en {val_file}")
            
            # Crear división de validación del conjunto principal
            torch.manual_seed(seed)
            train_size = int(len(math_indices) * (1 - val_split))
            train_indices = math_indices[:train_size]
            val_indices = math_indices[train_size:]
            
            # Crear subconjuntos
            train_subset = torch.utils.data.Subset(full_dataset, train_indices)
            val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        else:
            print(f"Encontrados {len(val_indices)} ejemplos matemáticos en {val_file}")
            
            # Usar conjunto de entrenamiento completo y conjunto de validación específico
            train_subset = torch.utils.data.Subset(full_dataset, math_indices)
            val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    else:
        # Crear división de validación del conjunto principal
        torch.manual_seed(seed)
        train_size = int(len(math_indices) * (1 - val_split))
        train_indices = math_indices[:train_size]
        val_indices = math_indices[train_size:]
        
        # Crear subconjuntos
        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Crear dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=math_collate_fn
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=math_collate_fn
    )
    
    print(f"DataLoader de entrenamiento: {len(train_subset)} ejemplos")
    print(f"DataLoader de validación: {len(val_subset)} ejemplos")
    
    return train_loader, val_loader

def train_math_expert():
    """Función principal de entrenamiento del experto matemático."""
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
    config = MathExpertConfig(
        input_dim=128,
        hidden_dim=256,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout
    )
    
    model = MathExpert(config)
    model.to(device)
    print(f"Modelo creado con {sum(p.numel() for p in model.parameters())} parámetros")
    
    # Guardar configuración
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        # Convertir config a dict
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
        json.dump(config_dict, f, indent=4)
    
    # Crear dataloaders
    train_loader, val_loader = create_math_dataloaders(
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
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )
    
    criterion = nn.MSELoss()
    
    # Inicializar variables para seguimiento
    best_val_loss = float('inf')
    start_time = time.time()
    history = {
        'train_loss': [],
        'val_loss': [],
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
            train_accuracy = 0.0
            train_samples = 0
            
            for batch_idx, (inputs, metadata) in enumerate(train_loader):
                inputs = inputs.to(device)
                target_result = metadata['result'].to(device)
                
                # Forward pass
                output, output_metadata = model(inputs)
                
                # Calcular pérdida
                loss = criterion(output_metadata['neural_result'], target_result)
                
                # Backward pass y optimización
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Actualizar estadísticas
                train_loss += loss.item() * inputs.size(0)
                train_samples += inputs.size(0)
                
                # Mostrar progreso
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch}/{args.epochs} - Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.6f}")
            
            # Calcular pérdida de entrenamiento promedio
            train_loss /= train_samples
            
            # Modo evaluación
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_samples = 0
            
            with torch.no_grad():
                for inputs, metadata in val_loader:
                    inputs = inputs.to(device)
                    target_result = metadata['result'].to(device)
                    target_operation = metadata['math_operation'].to(device)
                    
                    # Forward pass
                    output, output_metadata = model(inputs)
                    
                    # Calcular pérdida
                    loss = criterion(output_metadata['neural_result'], target_result)
                    
                    # Actualizar estadísticas
                    val_loss += loss.item() * inputs.size(0)
                    val_samples += inputs.size(0)
                    
                    # Calcular precisión de operación
                    predicted_op = output_metadata['operation']
                    val_correct += (predicted_op == target_operation).sum().item()
            
            # Calcular pérdida y precisión de validación promedio
            val_loss /= val_samples
            val_accuracy = val_correct / val_samples
            
            # Actualizar learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Guardar estadísticas en historial
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
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
            print(f"  Precisión operación: {val_accuracy:.4f}")
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
    train_math_expert()