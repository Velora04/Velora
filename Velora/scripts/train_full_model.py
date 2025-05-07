#!/usr/bin/env python
"""
Script para realizar el fine-tuning completo de VELORA.

Este script carga un modelo VELORA pre-entrenado con expertos, router y fusión,
y realiza un fine-tuning gradual de todo el sistema para optimizar la interacción
entre componentes.
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# Asegurar que el módulo raíz esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.velora import VELORA
from config.model_config import ModelConfig
from csv_dataloader import create_dataloaders_from_csv

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Fine-tuning del modelo VELORA completo')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Ruta al modelo pre-entrenado de VELORA (router y fusión)')
    
    parser.add_argument('--data_file', type=str, default='data/combined_dataset.csv',
                      help='Archivo CSV con datos mixtos para entrenamiento')
    
    parser.add_argument('--val_file', type=str, default=None,
                      help='Archivo CSV con datos mixtos para validación (opcional)')
    
    parser.add_argument('--output_dir', type=str, default='models/velora',
                      help='Directorio donde guardar los modelos entrenados')
    
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Tamaño del batch para entrenamiento')
    
    parser.add_argument('--epochs', type=int, default=30,
                      help='Número de épocas de entrenamiento')
    
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Tasa de aprendizaje inicial')
    
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                      help='Factor de weight decay para el optimizador')
    
    parser.add_argument('--save_interval', type=int, default=5,
                      help='Guardar modelo cada N épocas')
    
    parser.add_argument('--no_cuda', action='store_true',
                      help='Deshabilitar uso de CUDA incluso si está disponible')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Semilla para generación de números aleatorios')
    
    return parser.parse_args()

def create_layer_specific_optimizer(model, lr_config):
    """
    Crea un optimizador con tasas de aprendizaje específicas por capa.
    
    Args:
        model: Modelo VELORA
        lr_config: Diccionario con configuración de tasas de aprendizaje por grupo
            
    Returns:
        Optimizador configurado
    """
    param_groups = []
    
    # Agrupar parámetros por tipo
    for group_name, group_lr in lr_config.items():
        params = []
        
        # Filtrar parámetros por nombre
        for name, param in model.named_parameters():
            if group_name in name and param.requires_grad:
                params.append(param)
        
        # Si hay parámetros en este grupo, agregar al optimizador
        if params:
            param_groups.append({
                'params': params,
                'lr': group_lr,
                'name': group_name
            })
            print(f"Grupo '{group_name}': {len(params)} parámetros con lr={group_lr}")
    
    # Crear optimizador
    optimizer = optim.AdamW(param_groups, weight_decay=1e-6)
    return optimizer

def train_full_model():
    """Función principal para el fine-tuning del modelo completo."""
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
    
    # Verificar si existe el modelo pre-entrenado
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No se encontró el modelo pre-entrenado: {args.model_path}")
    
    # Cargar configuración del modelo
    config = ModelConfig()
    
    # Crear modelo
    print("Inicializando modelo VELORA...")
    model = VELORA(config)
    
    # Cargar pesos pre-entrenados
    print(f"Cargando pesos pre-entrenados desde: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    
    # Mover modelo a dispositivo
    model.to(device)
    
    # Crear dataloader
    # Si no se proporciona archivo de validación, usar el mismo que entrenamiento
    val_file = args.val_file if args.val_file else args.data_file
    
    train_loader, val_loader = create_dataloaders_from_csv(
        train_file=args.data_file,
        val_file=val_file,
        batch_size=args.batch_size,
        input_dim=config.input_dim
    )
    
    # Definir fases de fine-tuning
    fine_tuning_phases = [
        {
            # Fase 1: Solo entrenar capas superiores del router y fusión
            'name': 'Fase 1: Capas superiores',
            'epochs': args.epochs // 3,
            'unfreeze_layers': ['router.domain_classifier', 'fusion.fusion_network'],
            'lr_config': {
                'router.domain_classifier': args.lr,
                'fusion.fusion_network': args.lr,
                'default': args.lr / 10
            }
        },
        {
            # Fase 2: Descongelar capas intermedias de expertos
            'name': 'Fase 2: Capas intermedias de expertos',
            'epochs': args.epochs // 3,
            'unfreeze_layers': [
                'experts.math.operation_networks',
                'experts.language.self_attention',
                'router.attention'
            ],
            'lr_config': {
                'router': args.lr,
                'fusion': args.lr,
                'experts.math.operation_networks': args.lr / 2,
                'experts.language.self_attention': args.lr / 2,
                'default': args.lr / 5
            }
        },
        {
            # Fase 3: Fine-tuning completo con baja tasa de aprendizaje
            'name': 'Fase 3: Modelo completo',
            'epochs': args.epochs // 3 + args.epochs % 3,  # Añadir resto
            'unfreeze_layers': ['experts'],  # Descongelar todos los expertos
            'lr_config': {
                'router': args.lr / 2,
                'fusion': args.lr / 2,
                'experts': args.lr / 10,
                'default': args.lr / 20
            }
        }
    ]
    
    # Criterios para diferentes aspectos
    domain_criterion = nn.CrossEntropyLoss()  # Para clasificación de dominio
    math_op_criterion = nn.CrossEntropyLoss()  # Para clasificación de operación matemática
    language_task_criterion = nn.CrossEntropyLoss()  # Para clasificación de tarea de lenguaje
    result_criterion = nn.MSELoss()  # Para resultados numéricos/vectoriales
    
    # Inicializar variables para seguimiento
    best_val_loss = float('inf')
    start_time = time.time()
    history = {
        'train_loss': [],
        'val_loss': [],
        'domain_accuracy': [],
        'val_domain_accuracy': [],
        'math_accuracy': [],
        'val_math_accuracy': [],
        'language_accuracy': [],
        'val_language_accuracy': []
    }
    
    # Ciclo de fine-tuning por fases
    print("\nComenzando fine-tuning del modelo completo...")
    epoch_global = 0
    
    try:
        for phase_idx, phase in enumerate(fine_tuning_phases):
            print(f"\n{'-'*50}")
            print(f"Iniciando {phase['name']} por {phase['epochs']} épocas")
            print(f"{'-'*50}")
            
            # Congelar todos los parámetros primero
            for param in model.parameters():
                param.requires_grad = False
            
            # Descongelar capas específicas para esta fase
            for layer_pattern in phase['unfreeze_layers']:
                for name, param in model.named_parameters():
                    if layer_pattern in name:
                        param.requires_grad = True
                        
            # Imprimir parámetros entrenables
            trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
            print(f"Parámetros entrenables en esta fase: {len(trainable_params)}")
            for param_name in trainable_params[:10]:  # Mostrar algunos ejemplos
                print(f"  - {param_name}")
            if len(trainable_params) > 10:
                print(f"  ... y {len(trainable_params) - 10} más")
            
            # Crear optimizador específico para esta fase
            optimizer = create_layer_specific_optimizer(model, phase['lr_config'])
            
            # Crear scheduler
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=phase['epochs'] // 3 + 1,
                T_mult=1,
                eta_min=1e-6
            )
            
            # Entrenamiento para esta fase
            for epoch in range(1, phase['epochs'] + 1):
                epoch_global += 1
                epoch_start_time = time.time()
                
                # Modo entrenamiento
                model.train()
                train_loss = 0.0
                domain_correct = 0
                math_op_correct = 0
                lang_task_correct = 0
                total_samples = 0
                total_math = 0
                total_lang = 0
                
                for batch_idx, (inputs, metadata) in enumerate(train_loader):
                    # Mover datos a dispositivo
                    inputs = inputs.to(device)
                    target_domain = metadata['domain'].to(device)
                    target_math_operation = metadata['math_operation'].to(device)
                    target_language_task = metadata['language_task'].to(device)
                    target_result = metadata['result'].to(device)
                    
                    # Forward pass
                    outputs, output_metadata = model(inputs)
                    
                    # Calcular pérdidas
                    # 1. Pérdida de clasificación de dominio
                    domain_logits = output_metadata['routing']['domain_logits']
                    domain_loss = domain_criterion(domain_logits, target_domain)
                    
                    # 2. Pérdida específica según el dominio
                    # Para muestras matemáticas
                    math_mask = (target_domain == 0)
                    math_samples = math_mask.sum().item()
                    
                    # Para muestras de lenguaje
                    lang_mask = (target_domain == 1)
                    lang_samples = lang_mask.sum().item()
                    
                    # Inicializar pérdidas específicas
                    math_loss = torch.tensor(0.0, device=device)
                    lang_loss = torch.tensor(0.0, device=device)
                    
                    # Si hay muestras matemáticas, calcular pérdida
                    if math_samples > 0:
                        math_op_logits = output_metadata['routing']['math_task_logits']
                        math_op_loss = math_op_criterion(
                            math_op_logits[math_mask],
                            target_math_operation[math_mask]
                        )
                        math_loss = math_op_loss
                    
                    # Si hay muestras de lenguaje, calcular pérdida
                    if lang_samples > 0:
                        lang_task_logits = output_metadata['routing']['language_task_logits']
                        lang_task_loss = language_task_criterion(
                            lang_task_logits[lang_mask],
                            target_language_task[lang_mask]
                        )
                        lang_loss = lang_task_loss
                    
                    # 3. Pérdida de fusión
                    fusion_loss = torch.tensor(0.0, device=device)
                    if 'consistency_score' in output_metadata['fusion']:
                        # Promover alta consistencia con los expertos
                        consistency = output_metadata['fusion']['consistency_score']
                        fusion_loss = (1.0 - consistency).mean()
                    
                    # Combinar pérdidas
                    loss = domain_loss + math_loss + lang_loss + fusion_loss
                    
                    # Backward pass y optimización
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Actualizar estadísticas
                    train_loss += loss.item() * inputs.size(0)
                    
                    # Calcular precisión de clasificación de dominio
                    predicted_domain = output_metadata['routing']['primary_domain']
                    domain_correct += (predicted_domain == target_domain).sum().item()
                    total_samples += inputs.size(0)
                    
                    # Calcular precisión por dominio
                    for i in range(len(target_domain)):
                        if target_domain[i].item() == 0:  # Matemáticas
                            total_math += 1
                            if predicted_domain[i].item() == 0:
                                # Verificar operación matemática
                                pred_op = output_metadata['routing']['math_operation'][i]
                                true_op = target_math_operation[i]
                                if pred_op == true_op:
                                    math_op_correct += 1
                        else:  # Lenguaje
                            total_lang += 1
                            if predicted_domain[i].item() == 1:
                                # Verificar tarea de lenguaje
                                pred_task = output_metadata['routing']['language_task'][i]
                                true_task = target_language_task[i]
                                if pred_task == true_task:
                                    lang_task_correct += 1
                    
                    # Mostrar progreso
                    if (batch_idx + 1) % 10 == 0:
                        print(f"Fase {phase_idx+1}/{len(fine_tuning_phases)} - "
                              f"Epoch {epoch}/{phase['epochs']} "
                              f"(Global: {epoch_global}/{args.epochs}) - "
                              f"Batch {batch_idx+1}/{len(train_loader)} - "
                              f"Loss: {loss.item():.6f}")
                
                # Actualizar scheduler
                scheduler.step()
                
                # Calcular pérdida y precisión de entrenamiento promedio
                train_loss /= total_samples
                domain_accuracy = domain_correct / total_samples if total_samples > 0 else 0
                math_accuracy = math_op_correct / total_math if total_math > 0 else 0
                lang_accuracy = lang_task_correct / total_lang if total_lang > 0 else 0
                
                # Modo evaluación
                model.eval()
                val_loss = 0.0
                val_domain_correct = 0
                val_math_op_correct = 0
                val_lang_task_correct = 0
                val_total_samples = 0
                val_total_math = 0
                val_total_lang = 0
                
                with torch.no_grad():
                    for inputs, metadata in val_loader:
                        # Mover datos a dispositivo
                        inputs = inputs.to(device)
                        target_domain = metadata['domain'].to(device)
                        target_math_operation = metadata['math_operation'].to(device)
                        target_language_task = metadata['language_task'].to(device)
                        target_result = metadata['result'].to(device)
                        
                        # Forward pass
                        outputs, output_metadata = model(inputs)
                        
                        # Calcular pérdidas (similar a entrenamiento)
                        domain_logits = output_metadata['routing']['domain_logits']
                        domain_loss = domain_criterion(domain_logits, target_domain)
                        
                        # Específicas por dominio
                        math_mask = (target_domain == 0)
                        math_samples = math_mask.sum().item()
                        
                        lang_mask = (target_domain == 1)
                        lang_samples = lang_mask.sum().item()
                        
                        math_loss = torch.tensor(0.0, device=device)
                        lang_loss = torch.tensor(0.0, device=device)
                        
                        if math_samples > 0:
                            math_op_logits = output_metadata['routing']['math_task_logits']
                            math_op_loss = math_op_criterion(
                                math_op_logits[math_mask],
                                target_math_operation[math_mask]
                            )
                            math_loss = math_op_loss
                        
                        if lang_samples > 0:
                            lang_task_logits = output_metadata['routing']['language_task_logits']
                            lang_task_loss = language_task_criterion(
                                lang_task_logits[lang_mask],
                                target_language_task[lang_mask]
                            )
                            lang_loss = lang_task_loss
                        
                        fusion_loss = torch.tensor(0.0, device=device)
                        if 'consistency_score' in output_metadata['fusion']:
                            consistency = output_metadata['fusion']['consistency_score']
                            fusion_loss = (1.0 - consistency).mean()
                        
                        loss = domain_loss + math_loss + lang_loss + fusion_loss
                        
                        # Actualizar estadísticas
                        val_loss += loss.item() * inputs.size(0)
                        
                        # Calcular precisión de clasificación de dominio
                        predicted_domain = output_metadata['routing']['primary_domain']
                        val_domain_correct += (predicted_domain == target_domain).sum().item()
                        val_total_samples += inputs.size(0)
                        
                        # Calcular precisión por dominio
                        for i in range(len(target_domain)):
                            if target_domain[i].item() == 0:  # Matemáticas
                                val_total_math += 1
                                if predicted_domain[i].item() == 0:
                                    # Verificar operación matemática
                                    pred_op = output_metadata['routing']['math_operation'][i]
                                    true_op = target_math_operation[i]
                                    if pred_op == true_op:
                                        val_math_op_correct += 1
                            else:  # Lenguaje
                                val_total_lang += 1
                                if predicted_domain[i].item() == 1:
                                    # Verificar tarea de lenguaje
                                    pred_task = output_metadata['routing']['language_task'][i]
                                    true_task = target_language_task[i]
                                    if pred_task == true_task:
                                        val_lang_task_correct += 1
                
                # Calcular pérdida y precisión de validación promedio
                val_loss /= val_total_samples if val_total_samples > 0 else 1
                val_domain_accuracy = val_domain_correct / val_total_samples if val_total_samples > 0 else 0
                val_math_accuracy = val_math_op_correct / val_total_math if val_total_math > 0 else 0
                val_lang_accuracy = val_lang_task_correct / val_total_lang if val_total_lang > 0 else 0
                
                # Guardar estadísticas en historial
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['domain_accuracy'].append(domain_accuracy)
                history['val_domain_accuracy'].append(val_domain_accuracy)
                history['math_accuracy'].append(math_accuracy)
                history['val_math_accuracy'].append(val_math_accuracy)
                history['language_accuracy'].append(lang_accuracy)
                history['val_language_accuracy'].append(val_lang_accuracy)
                
                # Guardar historial en cada época
                with open(os.path.join(args.output_dir, "full_model_history.json"), 'w') as f:
                    json.dump(history, f, indent=4)
                
                # Calcular tiempo de época
                epoch_time = time.time() - epoch_start_time
                
                # Mostrar estadísticas de época
                print(f"Fase {phase_idx+1}/{len(fine_tuning_phases)} - "
                      f"Epoch {epoch}/{phase['epochs']} "
                      f"(Global: {epoch_global}/{args.epochs}) - "
                      f"Tiempo: {epoch_time:.2f}s")
                print(f"  Train Loss: {train_loss:.6f}, Domain Acc: {domain_accuracy:.4f}")
                print(f"  Math Acc: {math_accuracy:.4f}, Lang Acc: {lang_accuracy:.4f}")
                print(f"  Val Loss: {val_loss:.6f}, Val Domain Acc: {val_domain_accuracy:.4f}")
                print(f"  Val Math Acc: {val_math_accuracy:.4f}, Val Lang Acc: {val_lang_accuracy:.4f}")
                
                # Guardar mejor modelo
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(args.output_dir, "full_model_best.pt")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"  Guardado mejor modelo con val_loss={val_loss:.6f}")
                
                # Guardar checkpoint periódico
                if epoch_global % args.save_interval == 0 or epoch_global == args.epochs:
                    checkpoint_path = os.path.join(args.output_dir, f"full_model_epoch_{epoch_global}.pt")
                    torch.save({
                        'epoch': epoch_global,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, checkpoint_path)
                    print(f"  Guardado checkpoint en época global {epoch_global}")
                
                # Verificar para siguiente fase
                if epoch == phase['epochs']:
                    print(f"Completada fase {phase_idx+1}/{len(fine_tuning_phases)}: {phase['name']}")
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario.")
    
    # Calcular tiempo total
    total_time = time.time() - start_time
    print(f"\nFine-tuning completado en {total_time:.2f} segundos")
    print(f"Mejor pérdida de validación: {best_val_loss:.6f}")
    
    # Guardar modelo final
    final_model_path = os.path.join(args.output_dir, "full_model_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Modelo final guardado en: {final_model_path}")
    
    return model, history

if __name__ == "__main__":
    train_full_model()