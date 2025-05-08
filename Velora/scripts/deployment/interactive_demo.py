#!/usr/bin/env python
"""
Script de demostración para VELORA.

Este script permite interactuar con el modelo VELORA,
probando diferentes tipos de consultas en modo interactivo.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import re

# Agregar directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.velora import VELORA
from config.model_configs.base_config import BaseConfig


def parse_args():
    """Analiza argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Demostración interactiva de VELORA')
    
    parser.add_argument('--model_path', type=str, required=True,
                      help='Ruta al modelo VELORA entrenado')
    
    parser.add_argument('--device', type=str, default='cuda',
                      help='Dispositivo para inferencia (cuda, cpu)')
    
    parser.add_argument('--interactive', action='store_true',
                      help='Modo interactivo para consultas continuas')
    
    parser.add_argument('--examples', action='store_true',
                      help='Mostrar ejemplos integrados')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Carga modelo VELORA desde checkpoint."""
    print(f"Cargando modelo desde {model_path}...")
    
    # Crear configuración base
    config = BaseConfig()
    
    # Crear modelo
    model = VELORA(config)
    
    # Cargar pesos
    checkpoint = torch.load(model_path, map_location=device)
    
    # Si es un checkpoint completo, extraer state_dict del modelo
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Si es directamente el state_dict
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print("Modelo cargado exitosamente.")
    return model


def process_input(model, text_input, device):
    """
    Procesa una entrada de texto a través de VELORA.
    
    En una implementación real, esto requeriría un tokenizador
    pero para esta demostración, usamos una función simulada.
    """
    # Simular tokenización
    def simulate_tokenization(text):
        # Esto es una simplificación extrema - en un sistema real
        # usaríamos un tokenizador pre-entrenado
        tokens = text.lower().split()
        token_ids = [hash(token) % 10000 for token in tokens]
        return torch.tensor([token_ids]), torch.ones(1, len(token_ids))
    
    # Detectar si la entrada parece matemática
    is_math = bool(re.search(r'[\d+\-*/=]', text_input))
    
    # Simulación de embedding
    def simulate_embedding(token_ids, is_math=False):
        # Esta es una simulación - en un sistema real calcularíamos
        # embeddings reales basados en un modelo pre-entrenado
        embedding_dim = model.hidden_dim
        batch_size, seq_len = token_ids.size()
        
        # Crear embeddings aleatorios pero deterministas basados en tokens
        embeddings = torch.zeros(batch_size, seq_len, embedding_dim)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # Usar token_id como semilla para generar vector "pseudo-determinista"
                np.random.seed(int(token_ids[b, s]))
                embeddings[b, s] = torch.tensor(np.random.randn(embedding_dim))
        
        # Si es matemática, dar un "hint" insertando patrones en el embedding
        if is_math:
            embeddings[:, 0, :10] = torch.tensor([1.0, 0.5, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        return embeddings
    
    # Obtener "tokens" y atención
    token_ids, attention_mask = simulate_tokenization(text_input)
    token_ids = token_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Generar "embeddings"
    hidden_states = simulate_embedding(token_ids, is_math)
    hidden_states = hidden_states.to(device)
    
    # Procesar a través del modelo
    with torch.no_grad():
        outputs, metadata = model(hidden_states, attention_mask)
    
    # Generar explicación
    explanation = model.explain(metadata)[0]
    
    return outputs, metadata, explanation


def generate_response(outputs, metadata, explanation):
    """
    Genera una respuesta basada en la salida del modelo.
    
    En una implementación real, esta función decodificaría
    el resultado en lenguaje natural usando un módulo específico.
    """
    # Extraer dominio principal
    primary_domain = metadata['primary_routing']['primary_domain'][0].item()
    domain_name = "aritmético" if primary_domain == 0 else "lenguaje"
    
    # Extraer confianza
    confidence = metadata['primary_routing']['confidence'][0].item()
    
    # Respuesta base
    response = f"[Dominio: {domain_name}, Confianza: {confidence:.2f}]\n\n"
    
    # Si es dominio aritmético
    if primary_domain == 0:
        # Extraer información específica
        if 'expert_metadata' in metadata and 'final_result' in metadata['expert_metadata'][0]:
            result = metadata['expert_metadata'][0]['final_result'].item()
            operation = metadata['expert_routing'][0]['operation'].item()
            
            op_names = ["suma", "resta", "multiplicación", "división"]
            op_name = op_names[operation]
            
            response += f"He detectado una operación de {op_name}.\n"
            response += f"El resultado calculado es: {result:.4f}"
    
    # Si es dominio lingüístico
    else:
        # Extraer información específica
        if 'expert_routing' in metadata and 'task' in metadata['expert_routing'][0]:
            task = metadata['expert_routing'][0]['task'].item()
            
            task_names = ["pregunta", "comando", "declaración"]
            task_name = task_names[task]
            
            response += f"He detectado una consulta de tipo {task_name}.\n"
            
            # Respuesta simulada basada en tipo
            if task == 0:  # Pregunta
                response += "Para responder a esta pregunta, necesitaría más contexto o conocimiento específico."
            elif task == 1:  # Comando
                response += "Entiendo que me pides realizar una acción específica. En una implementación completa, ejecutaría el comando solicitado."
            else:  # Declaración
                response += "He procesado tu declaración. En una implementación completa, podría verificar o complementar esta información."
    
    # Añadir explicación del procesamiento interno
    response += f"\n\n[Explicación interna del modelo]\n{explanation}"
    
    return response


def show_examples():
    """Muestra ejemplos de consultas que el modelo puede procesar."""
    print("\nEjemplos de consultas que puedes probar:")
    
    print("\nOperaciones Aritméticas:")
    print("  • Calcula 25 + 18")
    print("  • ¿Cuánto es 145 - 89?")
    print("  • Multiplica 12 por 7")
    print("  • Divide 128 entre 4")
    
    print("\nPreguntas:")
    print("  • ¿Cómo funciona un algoritmo de búsqueda binaria?")
    print("  • ¿Cuáles son los componentes principales de VELORA?")
    print("  • ¿Por qué es importante la modularidad en sistemas de IA?")
    
    print("\nComandos:")
    print("  • Explica la arquitectura del modelo en detalle")
    print("  • Muestra un ejemplo de código para el enrutador")
    print("  • Genera una descripción del experto aritmético")
    
    print("\nDeclaraciones:")
    print("  • La especialización modular mejora el rendimiento en tareas específicas")
    print("  • Los sistemas de memoria son esenciales para mantener contexto")
    print("  • El enrutamiento adaptativo permite combinar diferentes capacidades")


def main():
    """Función principal de la demostración."""
    args = parse_args()
    
    # Determinar dispositivo
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar modelo
    model = load_model(args.model_path, device)
    
    # Mostrar ejemplos si se solicita
    if args.examples:
        show_examples()
    
    if args.interactive:
        print("\n" + "="*50)
        print("Demostración Interactiva de VELORA")
        print("="*50)
        print("Escribe cualquier consulta aritmética o lingüística.")
        print("Escribe 'salir' para terminar la demostración.\n")
        
        # Iniciar sesión si el modelo tiene gestor de contexto
        if hasattr(model, 'context_manager') and model.context_manager is not None:
            model.context_manager.start_session(1, device)
        
        # Bucle de interacción
        while True:
            try:
                user_input = input("\n> ")
                
                if user_input.lower() in ['salir', 'exit', 'quit', 'q']:
                    break
                
                if not user_input.strip():
                    continue
                
                # Procesar entrada
                outputs, metadata, explanation = process_input(model, user_input, device)
                
                # Generar respuesta
                response = generate_response(outputs, metadata, explanation)
                
                # Mostrar respuesta
                print("\nRespuesta:")
                print(response)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error al procesar consulta: {e}")
        
        print("\nFinalizando demostración.")
        
        # Finalizar sesión si corresponde
        if hasattr(model, 'context_manager') and model.context_manager is not None:
            model.context_manager.end_session()
    
    else:
        # Si no es interactivo, mostrar instrucciones
        print("\nEjecuta con --interactive para iniciar modo interactivo")
        print("o con --examples para ver ejemplos de consultas.")


if __name__ == "__main__":
    main()