#!/usr/bin/env python
"""
Script para entrenamiento completo de VELORA.

Este script maneja el proceso de fine-tuning integrado de VELORA,
coordinando múltiples componentes y fases de entrenamiento.
"""
import os
import sys
import argparse
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.models.velora import VELORA
from config.model_configs.base_config import BaseConfig
from src.training.objectives.integration_losses import VeloraLoss


def parse_args():
    """Analiza argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenamiento del modelo VELORA completo')
    
    parser.add_argument('--math_expert', type=str, required=True,
                      help='Ruta al experto matemático pre-entrenado')
    
    parser.add_argument('--language_expert', type=str, required=True,
                      help='Ruta al experto de lenguaje pre-entrenado')
    
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directorio con datos de entrenamiento')
    
    parser.add_argument('--output_dir', type=str, default='checkpoints/velora',
                      help='Directorio para guardar checkpoints')
    
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Tamaño de batch para entrenamiento')
    
    parser.add_argument('--epochs', type=int, default=30,
                      help='Número total de epochs')
    
    parser.add_argument('--lr', type=float, default=5e-5,
                      help='Tasa de aprendizaje inicial')
    
    parser.add_argument('--phases', type=int, default=3,
                      help='Número de fases de entrenamiento')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Semilla para reproducibilidad')
    
    parser.add_argument('--freeze_experts', action='store_true',
                      help='Congelar expertos durante el entrenamiento inicial')
    
    parser.add_argument('--device', type=str, default='cuda',
                      help='Dispositivo para entrenamiento (cuda, cpu)')
    
    parser.add_argument('--log_steps', type=int, default=100,
                      help='Frecuencia de logging durante entrenamiento')
    
    return parser.parse_args()


def setup_logging(output_dir):
    """Configura sistema de logging."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"training_{timestamp}.log")
    
    # Configurar logging básico
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("VELORA")


def create_loss_function(config):
    """Crea función de pérdida compuesta para entrenamiento."""
    return VeloraLoss(config)


def create_optimizer(model, config, freeze_experts=True):
    """
    Crea optimizador con diferentes tasas de aprendizaje
    para distintos componentes del modelo.
    """
    # Si expertos congelados, excluir sus parámetros
    if freeze_experts:
        model.freeze_experts()
    
    # Separar parámetros por componente para diferentes lr
    router_params = []
    fusion_params = []
    memory_params = []
    expert_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if 'router' in name:
            router_params.append(param)
        elif 'fusion' in name:
            fusion_params.append(param)
        elif 'memory' in name or 'context' in name:
            memory_params.append(param)
        else:
            expert_params.append(param)
    
    # Crear grupos de parámetros con diferentes lr
    param_groups = [
        {'params': router_params, 'lr': config.learning_rate},
        {'params': fusion_params, 'lr': config.learning_rate},
        {'params': memory_params, 'lr': config.learning_rate * 0.8},
        {'params': expert_params, 'lr': config.learning_rate * 0.1}
    ]
    
    # Crear optimizador
    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    
    return optimizer


def create_scheduler(optimizer, config, num_epochs):
    """Crea scheduler para ajustar tasa de aprendizaje."""
    return CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=max(1, num_epochs // 3),  # Un tercio de epochs
        T_mult=2,
        eta_min=config.learning_rate * 0.01
    )


def load_experts(math_expert_path, language_expert_path, config, device):
    """Carga expertos pre-entrenados."""
    # Crear modelos expertos iniciales
    from src.models.experts.arithmetic.math_expert import MathExpert
    from src.models.experts.language.language_expert import LanguageExpert
    
    # Inicializar expertos
    math_expert = MathExpert(config)
    language_expert = LanguageExpert(config)
    
    # Cargar pesos pre-entrenados
    logger.info(f"Cargando experto matemático desde {math_expert_path}")
    math_expert.load_state_dict(
        torch.load(math_expert_path, map_location=device)
    )
    
    logger.info(f"Cargando experto de lenguaje desde {language_expert_path}")
    language_expert.load_state_dict(
        torch.load(language_expert_path, map_location=device)
    )
    
    return math_expert, language_expert


def load_datasets(data_dir, batch_size, config):
    """Carga datasets para entrenamiento y validación."""
    # Esta función debería implementarse para cargar los datos específicos de entrenamiento
    # Aquí presentamos una implementación simplificada como ejemplo
    
    from src.data.datasets.velora_dataset import VeloraDataset, velora_collate_fn
    
    train_dataset = VeloraDataset(
        os.path.join(data_dir, 'train.csv'),
        config
    )
    
    val_dataset = VeloraDataset(
        os.path.join(data_dir, 'val.csv'),
        config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=velora_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=velora_collate_fn
    )
    
    return train_loader, val_loader


def train_epoch(model, dataloader, optimizer, loss_fn, device, log_steps, logger):
    """Entrena el modelo por una época."""
    model.train()
    total_loss = 0
    total_samples = 0
    start_time = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        # Extraer datos y moverlos a dispositivo
        inputs = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        domain_labels = batch['domain'].to(device)
        math_op_labels = batch['math_operation'].to(device)
        language_task_labels = batch['language_task'].to(device)
        result_labels = batch['result'].to(device)
        
        # Forward pass
        outputs, metadata = model(inputs, attention_mask)
        
        # Calcular pérdida
        loss_dict = loss_fn(
            outputs=outputs,
            metadata=metadata,
            domain_labels=domain_labels,
            math_op_labels=math_op_labels,
            language_task_labels=language_task_labels,
            result_labels=result_labels
        )
        
        # Extraer pérdida total
        loss = loss_dict['total']
        
        # Backward y optimización
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping opcional
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Actualizar estadísticas
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Log periódico
        if (batch_idx + 1) % log_steps == 0:
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Domain: {loss_dict['domain']:.4f} | "
                f"Tasks: {loss_dict['tasks']:.4f} | "
                f"Results: {loss_dict['results']:.4f} | "
                f"LR: {lr:.6f}"
            )
    
    # Calcular métricas finales
    epoch_loss = total_loss / total_samples
    epoch_time = time.time() - start_time
    
    return epoch_loss, epoch_time


def evaluate(model, dataloader, loss_fn, device):
    """Evalúa el modelo en dataset de validación."""
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # Métricas adicionales
    domain_correct = 0
    math_op_correct = 0
    language_task_correct = 0
    math_samples = 0
    language_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Extraer datos y moverlos a dispositivo
            inputs = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            domain_labels = batch['domain'].to(device)
            math_op_labels = batch['math_operation'].to(device)
            language_task_labels = batch['language_task'].to(device)
            result_labels = batch['result'].to(device)
            
            # Forward pass
            outputs, metadata = model(inputs, attention_mask)
            
            # Calcular pérdida
            loss_dict = loss_fn(
                outputs=outputs,
                metadata=metadata,
                domain_labels=domain_labels,
                math_op_labels=math_op_labels,
                language_task_labels=language_task_labels,
                result_labels=result_labels
            )
            
            # Extraer pérdida total
            loss = loss_dict['total']
            
            # Actualizar estadísticas
            batch_size = inputs.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Evaluar predicciones de dominio
            primary_domain = metadata['primary_routing']['primary_domain']
            domain_correct += (primary_domain == domain_labels).sum().item()
            
            # Evaluar predicciones específicas por dominio
            for i in range(batch_size):
                if domain_labels[i] == 0:  # Aritmética
                    math_samples += 1
                    if primary_domain[i] == 0:
                        pred_op = metadata['expert_routing'][i]['operation']
                        if pred_op == math_op_labels[i]:
                            math_op_correct += 1
                else:  # Lenguaje
                    language_samples += 1
                    if primary_domain[i] == 1:
                        pred_task = metadata['expert_routing'][i]['task']
                        if pred_task == language_task_labels[i]:
                            language_task_correct += 1
    
    # Calcular métricas finales
    val_loss = total_loss / total_samples
    domain_accuracy = domain_correct / total_samples if total_samples > 0 else 0
    math_op_accuracy = math_op_correct / math_samples if math_samples > 0 else 0
    language_task_accuracy = language_task_correct / language_samples if language_samples > 0 else 0
    
    return {
        'loss': val_loss,
        'domain_accuracy': domain_accuracy,
        'math_operation_accuracy': math_op_accuracy,
        'language_task_accuracy': language_task_accuracy
    }


def save_checkpoint(model, optimizer, scheduler, epoch, output_dir, metrics, name=None):
    """Guarda checkpoint del modelo."""
    if name is None:
        filename = f"velora_epoch_{epoch:03d}.pt"
    else:
        filename = f"{name}.pt"
    
    path = os.path.join(output_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }, path)
    
    logger.info(f"Checkpoint guardado en {path}")
    return path


def train_phase(
    phase_num,
    model,
    train_loader,
    val_loader,
    config,
    num_epochs,
    output_dir,
    device,
    log_steps,
    unfreeze_layers=None
):
    """Ejecuta una fase específica de entrenamiento."""
    logger.info(f"Iniciando Fase {phase_num}")
    
    # Si se especifican capas a descongelar
    if unfreeze_layers:
        logger.info(f"Descongelando capas: {unfreeze_layers}")
        model.unfreeze_layers(unfreeze_layers)
    
    # Crear optimizador y scheduler
    optimizer = create_optimizer(model, config, False)  # No congelar adicionalmente
    scheduler = create_scheduler(optimizer, config, num_epochs)
    
    # Crear función de pérdida
    loss_fn = create_loss_function(config)
    
    # Tracking de mejor modelo
    best_val_loss = float('inf')
    best_checkpoint_path = None
    
    # Historia de entrenamiento
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    # Ciclo de entrenamiento para esta fase
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Fase {phase_num} - Epoch {epoch}/{num_epochs}")
        
        # Entrenamiento
        train_loss, epoch_time = train_epoch(
            model, train_loader, optimizer, loss_fn, device, log_steps, logger
        )
        
        logger.info(
            f"Fase {phase_num} - Epoch {epoch} completada en {epoch_time:.2f}s | "
            f"Train Loss: {train_loss:.4f}"
        )
        
        # Evaluación
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        val_loss = val_metrics['loss']
        
        logger.info(
            f"Validation: Loss: {val_loss:.4f} | "
            f"Domain Acc: {val_metrics['domain_accuracy']:.4f} | "
            f"Math Op Acc: {val_metrics['math_operation_accuracy']:.4f} | "
            f"Lang Task Acc: {val_metrics['language_task_accuracy']:.4f}"
        )
        
        # Actualizar scheduler
        scheduler.step()
        
        # Guardar historia
        history['train_loss'].append(train_loss)
        history['val_metrics'].append(val_metrics)
        
        # Guardar checkpoint periódicamente
        if epoch % 5 == 0 or epoch == num_epochs:
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                output_dir, val_metrics, f"phase{phase_num}_epoch{epoch}"
            )
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = save_checkpoint(
                model, optimizer, scheduler, epoch, 
                output_dir, val_metrics, f"phase{phase_num}_best"
            )
            logger.info(f"Nuevo mejor modelo guardado con val_loss={val_loss:.4f}")
    
    return history, best_checkpoint_path


def main():
    """Función principal de entrenamiento."""
    args = parse_args()
    
    # Configurar reproducibilidad
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Asegurar que directorios existan
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configurar logging
    global logger
    logger = setup_logging(args.output_dir)
    
    # Guardar configuración
    config_path = os.path.join(args.output_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Determinar dispositivo
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Utilizando dispositivo: {device}")
    
    # Cargar configuración
    config = BaseConfig()
    config.learning_rate = args.lr
    config.batch_size = args.batch_size
    
    # Cargar expertos pre-entrenados
    math_expert, language_expert = load_experts(
        args.math_expert, args.language_expert, config, device
    )
    
    # Crear modelo VELORA
    model = VELORA(
        config=config,
        math_expert=math_expert,
        language_expert=language_expert,
        freeze_experts=args.freeze_experts
    )
    model.to(device)
    
    # Cargar datasets
    train_loader, val_loader = load_datasets(
        args.data_dir, args.batch_size, config
    )
    
    logger.info(f"Dataset de entrenamiento: {len(train_loader.dataset)} muestras")
    logger.info(f"Dataset de validación: {len(val_loader.dataset)} muestras")
    
    # Dividir epochs entre fases
    phase_epochs = [args.epochs // args.phases] * args.phases
    # Distribuir epochs restantes
    remainder = args.epochs % args.phases
    for i in range(remainder):
        phase_epochs[i] += 1
    
    logger.info(f"Distribución de epochs por fase: {phase_epochs}")
    
    # Definir configuración de fases
    phase_configs = [
        {
            "unfreeze_layers": ["router", "fusion"],
            "description": "Entrenamiento de enrutador y fusión"
        },
        {
            "unfreeze_layers": ["router", "fusion", "experts.arithmetic.operation_networks"],
            "description": "Refinamiento de experto aritmético"
        },
        {
            "unfreeze_layers": None,  # Todo el modelo
            "description": "Fine-tuning completo"
        }
    ]
    
    # Entrenar por fases
    all_history = []
    best_checkpoints = []
    
    for phase in range(args.phases):
        logger.info(f"\n{'='*50}")
        logger.info(f"FASE {phase+1}/{args.phases}: {phase_configs[phase]['description']}")
        logger.info(f"{'='*50}\n")
        
        # Ejecutar fase de entrenamiento
        history, best_checkpoint = train_phase(
            phase_num=phase+1,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            num_epochs=phase_epochs[phase],
            output_dir=args.output_dir,
            device=device,
            log_steps=args.log_steps,
            unfreeze_layers=phase_configs[phase]['unfreeze_layers']
        )
        
        all_history.append(history)
        best_checkpoints.append(best_checkpoint)
    
    # Guardar historia completa
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump({
            'phase_epochs': phase_epochs,
            'phase_configs': phase_configs,
            'history': all_history,
            'best_checkpoints': best_checkpoints
        }, f, indent=4)
    
    # Guardar modelo final
    final_path = save_checkpoint(
        model, None, None, args.epochs, 
        args.output_dir, {}, "velora_final"
    )
    
    logger.info(f"\nEntrenamiento completo. Modelo final guardado en {final_path}")


if __name__ == "__main__":
    main()