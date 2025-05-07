#!/usr/bin/env python
"""
Script para evaluar el modelo VELORA completo.

Este script carga un modelo VELORA entrenado y evalúa su rendimiento
en diferentes tipos de tareas (matemáticas y lenguaje).
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Asegurar que el módulo raíz esté en el path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.velora import VELORA
from config.model_config import ModelConfig
from csv_dataloader import VeloraCSVDataset, create_dataloaders_from_csv

def parse_args():
    """Analiza los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Evaluación del modelo VELORA')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Ruta al modelo VELORA para evaluar')
    
    parser.add_argument('--test_file', type=str, default='data/velora_test.csv',
                      help='Archivo CSV con datos de prueba')
    
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Directorio donde guardar los resultados de evaluación')
    
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Tamaño del batch para evaluación')
    
    parser.add_argument('--detailed', action='store_true',
                      help='Realizar evaluación detallada por operación/tarea')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Generar visualizaciones de resultados')
    
    parser.add_argument('--examples', type=int, default=5,
                      help='Número de ejemplos de cada tipo para mostrar')
    
    parser.add_argument('--no_cuda', action='store_true',
                      help='Deshabilitar uso de CUDA incluso si está disponible')
    
    parser.add_argument('--seed', type=int, default=42,
                      help='Semilla para generación de números aleatorios')
    
    return parser.parse_args()

def evaluate_model():
    """Función principal de evaluación del modelo."""
    # Parsear argumentos
    args = parse_args()
    
    # Establecer semilla para reproducibilidad
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Configurar dispositivo
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Verificar si existe el modelo
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"No se encontró el modelo: {args.model_path}")
    
    # Verificar si existe el archivo de prueba
    if not os.path.exists(args.test_file):
        raise FileNotFoundError(f"No se encontró el archivo de prueba: {args.test_file}")
    
    # Cargar configuración del modelo
    config = ModelConfig()
    
    # Crear modelo
    print("Inicializando modelo VELORA...")
    model = VELORA(config)
    
    # Cargar pesos
    print(f"Cargando pesos del modelo desde: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    
    # Mover modelo a dispositivo y modo evaluación
    model.to(device)
    model.eval()
    
    # Cargar datos de prueba
    test_dataset = VeloraCSVDataset(args.test_file, input_dim=config.input_dim)
    
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
            batch_inputs = torch.zeros(batch_size, max_len, config.input_dim)
            
            for i, input_data in enumerate(inputs):
                seq_len = input_data.size(0)
                batch_inputs[i, :seq_len, :] = input_data
        else:
            # Para datos no secuenciales (matemáticas)
            batch_inputs = torch.stack(inputs)
        
        # Procesar metadatas
        batch_metadata = {}
        keys = metadatas[0].keys()
        
        for key in keys:
            if isinstance(metadatas[0][key], torch.Tensor):
                # Concatenar tensores
                batch_metadata[key] = torch.cat([m[key] for m in metadatas])
        
        return batch_inputs, batch_metadata
    
    # Crear dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print(f"Dataset de prueba cargado: {len(test_dataset)} muestras")
    
    # Estructuras para almacenar resultados
    results = {
        'samples': len(test_dataset),
        'domain': {
            'correct': 0,
            'total': len(test_dataset),
            'accuracy': 0
        },
        'math': {
            'correct': 0,
            'total': 0,
            'accuracy': 0,
            'operations': {
                0: {'name': 'suma', 'correct': 0, 'total': 0, 'accuracy': 0},
                1: {'name': 'resta', 'correct': 0, 'total': 0, 'accuracy': 0},
                2: {'name': 'multiplicación', 'correct': 0, 'total': 0, 'accuracy': 0},
                3: {'name': 'división', 'correct': 0, 'total': 0, 'accuracy': 0}
            }
        },
        'language': {
            'correct': 0,
            'total': 0,
            'accuracy': 0,
            'tasks': {
                0: {'name': 'pregunta', 'correct': 0, 'total': 0, 'accuracy': 0},
                1: {'name': 'comando', 'correct': 0, 'total': 0, 'accuracy': 0},
                2: {'name': 'declaración', 'correct': 0, 'total': 0, 'accuracy': 0}
            }
        }
    }
    
    # Variables para ejemplos representativos
    math_examples = {op: [] for op in range(4)}
    language_examples = {task: [] for task in range(3)}
    
    # Realizar evaluación
    print("\nEvaluando modelo...")
    total_loss = 0.0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_idx, (inputs, metadata) in enumerate(tqdm(test_loader, desc="Evaluando")):
            # Mover datos a dispositivo
            inputs = inputs.to(device)
            target_domain = metadata['domain'].to(device)
            target_math_operation = metadata['math_operation'].to(device)
            target_language_task = metadata['language_task'].to(device)
            target_result = metadata['result'].to(device)
            
            # Forward pass
            outputs, output_metadata = model(inputs)
            
            # Calcular pérdida
            loss = criterion(outputs, target_result)
            total_loss += loss.item() * inputs.size(0)
            
            # Evaluar predicciones
            predicted_domain = output_metadata['routing']['primary_domain']
            
            # Actualizar estadísticas de dominio
            correct_domain = (predicted_domain == target_domain).cpu().numpy()
            results['domain']['correct'] += correct_domain.sum()
            
            # Obtener explicaciones para ejemplos representativos
            explanations = model.explain(output_metadata)
            
            # Procesar cada muestra en el batch
            for i in range(len(target_domain)):
                true_domain = target_domain[i].item()
                pred_domain = predicted_domain[i].item()
                
                # Analizar por dominio
                if true_domain == 0:  # Matemáticas
                    results['math']['total'] += 1
                    true_op = target_math_operation[i].item()
                    results['math']['operations'][true_op]['total'] += 1
                    
                    if pred_domain == 0:
                        # Verificar operación matemática
                        pred_op = output_metadata['routing']['math_operation'][i].item()
                        
                        if pred_op == true_op:
                            results['math']['correct'] += 1
                            results['math']['operations'][true_op]['correct'] += 1
                        
                        # Guardar ejemplo si es interesante
                        # (correcto o incorrecto con alta confianza)
                        if len(math_examples[true_op]) < args.examples:
                            confidence = output_metadata['routing']['confidence'][i].item()
                            is_correct = (pred_op == true_op) and (pred_domain == true_domain)
                            
                            example = {
                                'input': inputs[i].cpu(),
                                'is_sequential': inputs[i].dim() > 1,
                                'true_domain': true_domain,
                                'pred_domain': pred_domain,
                                'true_operation': true_op,
                                'pred_operation': pred_op,
                                'confidence': confidence,
                                'is_correct': is_correct,
                                'explanation': explanations[i]
                            }
                            
                            math_examples[true_op].append(example)
                        
                else:  # Lenguaje
                    results['language']['total'] += 1
                    true_task = target_language_task[i].item()
                    results['language']['tasks'][true_task]['total'] += 1
                    
                    if pred_domain == 1:
                        # Verificar tarea de lenguaje
                        pred_task = output_metadata['routing']['language_task'][i].item()
                        
                        if pred_task == true_task:
                            results['language']['correct'] += 1
                            results['language']['tasks'][true_task]['correct'] += 1
                        
                        # Guardar ejemplo si es interesante
                        if len(language_examples[true_task]) < args.examples:
                            confidence = output_metadata['routing']['confidence'][i].item()
                            is_correct = (pred_task == true_task) and (pred_domain == true_domain)
                            
                            example = {
                                'input': inputs[i].cpu(),
                                'is_sequential': inputs[i].dim() > 1,
                                'true_domain': true_domain,
                                'pred_domain': pred_domain,
                                'true_task': true_task,
                                'pred_task': pred_task,
                                'confidence': confidence,
                                'is_correct': is_correct,
                                'explanation': explanations[i]
                            }
                            
                            language_examples[true_task].append(example)
    
    # Calcular métricas agregadas
    avg_loss = total_loss / len(test_dataset)
    results['loss'] = avg_loss
    
    # Calcular precisiones
    results['domain']['accuracy'] = results['domain']['correct'] / results['domain']['total'] if results['domain']['total'] > 0 else 0
    
    if results['math']['total'] > 0:
        results['math']['accuracy'] = results['math']['correct'] / results['math']['total']
        for op in results['math']['operations']:
            op_total = results['math']['operations'][op]['total']
            if op_total > 0:
                results['math']['operations'][op]['accuracy'] = results['math']['operations'][op]['correct'] / op_total
    
    if results['language']['total'] > 0:
        results['language']['accuracy'] = results['language']['correct'] / results['language']['total']
        for task in results['language']['tasks']:
            task_total = results['language']['tasks'][task]['total']
            if task_total > 0:
                results['language']['tasks'][task]['accuracy'] = results['language']['tasks'][task]['correct'] / task_total
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS DE EVALUACIÓN")
    print("="*50)
    
    print(f"\nMuestras totales: {results['samples']}")
    print(f"Pérdida promedio: {avg_loss:.6f}")
    print(f"Precisión de clasificación de dominio: {results['domain']['accuracy']:.4f} ({results['domain']['correct']}/{results['domain']['total']})")
    
    print("\nDominio Matemático:")
    math_acc = results['math']['accuracy']
    print(f"  Precisión global: {math_acc:.4f} ({results['math']['correct']}/{results['math']['total']})")
    
    if args.detailed:
        print("\n  Por operación:")
        for op in sorted(results['math']['operations'].keys()):
            op_data = results['math']['operations'][op]
            print(f"    {op_data['name']}: {op_data['accuracy']:.4f} ({op_data['correct']}/{op_data['total']})")
    
    print("\nDominio de Lenguaje:")
    lang_acc = results['language']['accuracy']
    print(f"  Precisión global: {lang_acc:.4f} ({results['language']['correct']}/{results['language']['total']})")
    
    if args.detailed:
        print("\n  Por tipo de consulta:")
        for task in sorted(results['language']['tasks'].keys()):
            task_data = results['language']['tasks'][task]
            print(f"    {task_data['name']}: {task_data['accuracy']:.4f} ({task_data['correct']}/{task_data['total']})")
    
    # Guardar resultados en JSON
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        # Convertir valores numpy a Python nativos para JSON
        json_results = json.dumps(results, default=lambda x: x.item() if hasattr(x, 'item') else x, indent=4)
        f.write(json_results)
    
    print(f"\nResultados guardados en: {results_file}")
    
    # Mostrar ejemplos si se solicita
    if args.detailed:
        print("\n" + "="*50)
        print("EJEMPLOS DE EVALUACIÓN")
        print("="*50)
        
        # Ejemplos matemáticos
        print("\nEjemplos Matemáticos:")
        for op in sorted(math_examples.keys()):
            op_name = results['math']['operations'][op]['name']
            print(f"\n  Operación: {op_name}")
            
            for i, example in enumerate(math_examples[op]):
                print(f"    Ejemplo {i+1}:")
                print(f"      Predicción: {'Correcta' if example['is_correct'] else 'Incorrecta'}")
                print(f"      Confianza: {example['confidence']:.4f}")
                print(f"      Explicación: {example['explanation']}")
                print()
        
        # Ejemplos de lenguaje
        print("\nEjemplos de Lenguaje:")
        for task in sorted(language_examples.keys()):
            task_name = results['language']['tasks'][task]['name']
            print(f"\n  Tarea: {task_name}")
            
            for i, example in enumerate(language_examples[task]):
                print(f"    Ejemplo {i+1}:")
                print(f"      Predicción: {'Correcta' if example['is_correct'] else 'Incorrecta'}")
                print(f"      Confianza: {example['confidence']:.4f}")
                print(f"      Explicación: {example['explanation']}")
                print()
    
    # Generar visualizaciones si se solicita
    if args.visualize:
        print("\nGenerando visualizaciones...")
        
        # Directorio para visualizaciones
        viz_dir = os.path.join(args.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Gráfico de precisión por dominio
        plt.figure(figsize=(10, 6))
        domains = ['General', 'Matemáticas', 'Lenguaje']
        accuracies = [results['domain']['accuracy'], results['math']['accuracy'], results['language']['accuracy']]
        colors = ['blue', 'green', 'red']
        
        plt.bar(domains, accuracies, color=colors)
        plt.ylim(0, 1.0)
        plt.ylabel('Precisión')
        plt.title('Precisión por Dominio')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.savefig(os.path.join(viz_dir, 'domain_accuracy.png'), dpi=300, bbox_inches='tight')
        
        # Gráfico detallado por operación matemática
        plt.figure(figsize=(10, 6))
        ops = [results['math']['operations'][op]['name'] for op in sorted(results['math']['operations'].keys())]
        op_accs = [results['math']['operations'][op]['accuracy'] for op in sorted(results['math']['operations'].keys())]
        
        plt.bar(ops, op_accs, color='green')
        plt.ylim(0, 1.0)
        plt.ylabel('Precisión')
        plt.title('Precisión por Operación Matemática')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(op_accs):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.savefig(os.path.join(viz_dir, 'math_operations_accuracy.png'), dpi=300, bbox_inches='tight')
        
        # Gráfico detallado por tarea de lenguaje
        plt.figure(figsize=(10, 6))
        tasks = [results['language']['tasks'][task]['name'] for task in sorted(results['language']['tasks'].keys())]
        task_accs = [results['language']['tasks'][task]['accuracy'] for task in sorted(results['language']['tasks'].keys())]
        
        plt.bar(tasks, task_accs, color='red')
        plt.ylim(0, 1.0)
        plt.ylabel('Precisión')
        plt.title('Precisión por Tipo de Consulta de Lenguaje')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        for i, v in enumerate(task_accs):
            plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
        
        plt.savefig(os.path.join(viz_dir, 'language_tasks_accuracy.png'), dpi=300, bbox_inches='tight')
        
        print(f"Visualizaciones guardadas en: {viz_dir}")
    
    return results

if __name__ == "__main__":
    evaluate_model()