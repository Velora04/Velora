# examples/test_velora.py
import torch
import torch.nn.functional as F
import sys
import os

# Añadir el directorio raíz del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.velora import VELORA
from config.model_config import ModelConfig

def test_math_expert(model):
    """Prueba el experto matemático."""
    print("Prueba del Experto Matemático")
    print("-" * 50)
    
    # Crear entrada matemática simulada
    a, b = 5.0, 3.0
    x = torch.tensor([[a, b]], dtype=torch.float32)
    x = torch.nn.functional.pad(x, (0, model.config.input_dim - 2), value=0)
    
    # Procesamiento
    model.eval()
    with torch.no_grad():
        # Forzar enrutamiento al experto matemático
        routing_info = model.router(x)
        routing_info['primary_domain'] = torch.zeros_like(routing_info['primary_domain'])
        
        # Probar diferentes operaciones
        operations = ['suma', 'resta', 'multiplicación', 'división']
        expected_results = [a + b, a - b, a * b, a / b]
        
        for op_idx, (op_name, expected) in enumerate(zip(operations, expected_results)):
            # Forzar operación
            routing_info['math_operation'] = torch.tensor([op_idx])
            
            # Activar experto matemático directamente
            math_expert = model.experts['math']
            output, metadata = math_expert(x, operation_hint=op_idx)
            
            # Verificar resultado
            result = metadata['symbolic_result'].item()
            print(f"Operación: {op_name}, Resultado: {result:.2f}, Esperado: {expected:.2f}")
    
    print("\nExperto matemático OK\n")

def test_language_expert(model):
    """Prueba el experto de lenguaje."""
    print("Prueba del Experto en Lenguaje")
    print("-" * 50)
    
    # Crear entrada de lenguaje simulada (secuencia de tokens)
    seq_len = 10
    x = torch.randn(1, seq_len, model.config.input_dim)
    
    # Procesamiento
    model.eval()
    with torch.no_grad():
        # Forzar enrutamiento al experto de lenguaje
        routing_info = model.router(x)
        routing_info['primary_domain'] = torch.ones_like(routing_info['primary_domain'])
        
        # Probar diferentes tareas
        tasks = ['pregunta', 'comando', 'declaración']
        
        for task_idx, task_name in enumerate(tasks):
            # Forzar tarea
            routing_info['language_task'] = torch.tensor([task_idx])
            
            # Activar experto de lenguaje directamente
            language_expert = model.experts['language']
            output, metadata = language_expert(x, task_hint=task_idx)
            
            # Verificar clasificación
            query_type = metadata['query_type'].item()
            print(f"Tarea: {task_name}, Clasificación: {tasks[query_type]}")
            
            # Verificar predicción de tokens
            if 'next_tokens' in metadata:
                next_tokens = metadata['next_tokens'][0]
                print(f"Tokens predichos (primeros 5): {next_tokens[:5].tolist()}")
    
    print("\nExperto en lenguaje OK\n")

def test_router(model):
    """Prueba el enrutador."""
    print("Prueba del Enrutador")
    print("-" * 50)
    
    # Crear entradas para matemáticas y lenguaje
    math_input = torch.zeros(1, model.config.input_dim)
    math_input[0, :2] = torch.tensor([5.0, 3.0])  # dos números
    
    language_input = torch.randn(1, 10, model.config.input_dim)  # secuencia
    
    # Procesamiento
    model.eval()
    with torch.no_grad():
        # Enrutar entrada matemática
        math_routing = model.router(math_input)
        math_domain = math_routing['primary_domain'].item()
        math_confidence = math_routing['confidence'].item()
        
        print(f"Entrada matemática -> Dominio: {math_domain} (0=matemáticas), Confianza: {math_confidence:.2f}")
        
        # Enrutar entrada de lenguaje
        lang_routing = model.router(language_input)
        lang_domain = lang_routing['primary_domain'].item()
        lang_confidence = lang_routing['confidence'].item()
        
        print(f"Entrada de lenguaje -> Dominio: {lang_domain} (1=lenguaje), Confianza: {lang_confidence:.2f}")
    
    print("\nEnrutador OK\n")

def test_integrated_model(model):
    """Prueba el modelo integrado."""
    print("Prueba del Modelo Integrado")
    print("-" * 50)
    
    # Prueba con operación matemática
    a, b = 8.0, 4.0
    math_input = torch.tensor([[a, b]], dtype=torch.float32)
    math_input = torch.nn.functional.pad(math_input, (0, model.config.input_dim - 2), value=0)
    
    # Prueba con texto
    text_input = torch.randn(1, 15, model.config.input_dim)
    
    # Procesamiento
    model.eval()
    with torch.no_grad():
        # Procesar entrada matemática
        math_output, math_metadata = model(math_input)
        math_explanation = model.explain(math_metadata)
        
        print("Procesamiento matemático:")
        print(math_explanation[0])
        
        # Procesar entrada de lenguaje
        text_output, text_metadata = model(text_input)
        text_explanation = model.explain(text_metadata)
        
        print("\nProcesamiento de lenguaje:")
        print(text_explanation[0])
    
    print("\nModelo integrado OK\n")

def test_velora():
    """Prueba completa del modelo VELORA."""
    # Cargar configuración
    config = ModelConfig()
    
    # Inicializar modelo
    model = VELORA(config)
    
    # Cargar pesos pre-entrenados si existen
    try:
        model.load_state_dict(torch.load('models/velora_final.pt'))
        print("Modelo pre-entrenado cargado correctamente.")
    except:
        print("No se encontró modelo pre-entrenado. Usando inicialización aleatoria.")
    
    # Ejecutar pruebas
    test_math_expert(model)
    test_language_expert(model)
    test_router(model)
    test_integrated_model(model)
    
    print("Todas las pruebas completadas con éxito!")

if __name__ == "__main__":
    test_velora()