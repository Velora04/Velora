# train_velora_menu.py
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import argparse
from datetime import datetime
import json
import glob

# Añadir la ruta del proyecto al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.velora_model import VeloraModel
from src.experts.math_expert import MathExpert
from src.experts.language_expert import LanguageExpert
from src.routing.router import VeloraRouter
from src.core.latent_space import SharedLatentSpace
from src.integration.fusion import OutputFusion

# Importar el cargador de dataset personalizado
from dataset_loader import create_data_loaders, VeloraCustomDataset

# Funciones de utilidad para la gestión de modelos
def list_available_models(models_dir="models"):
    """Lista todos los modelos guardados en el directorio."""
    if not os.path.exists(models_dir):
        return []
    
    model_files = glob.glob(os.path.join(models_dir, "*.pt"))
    return [os.path.basename(f) for f in model_files]

def save_training_config(config, filename="config/last_training.json"):
    """Guarda la configuración de entrenamiento en un archivo JSON."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Configuración guardada en {filename}")

def load_training_config(filename="config/last_training.json"):
    """Carga la configuración de entrenamiento desde un archivo JSON."""
    if not os.path.exists(filename):
        return None
    
    with open(filename, 'r') as f:
        config = json.load(f)
    
    print(f"Configuración cargada desde {filename}")
    return config

def list_available_datasets(data_dir="data"):
    """Lista todos los datasets disponibles."""
    if not os.path.exists(data_dir):
        return []
    
    dataset_files = glob.glob(os.path.join(data_dir, "*.csv"))
    return [os.path.basename(f) for f in dataset_files]

def get_model_info(model_path):
    """Obtiene información sobre un modelo guardado."""
    if not os.path.exists(model_path):
        return "Archivo no encontrado"
    
    file_stats = os.stat(model_path)
    size_mb = file_stats.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    
    return f"Tamaño: {size_mb:.2f} MB, Modificado: {mod_time}"

# Función principal de entrenamiento
def train_velora(config):
    """
    Entrena el modelo VELORA con la configuración proporcionada.
    
    Args:
        config: Diccionario con la configuración de entrenamiento
    """
    print("\n" + "="*50)
    print("Iniciando entrenamiento de VELORA...")
    print("="*50)
    
    # Extraer parámetros
    train_file = config['train_file']
    val_file = config['val_file']
    epochs = config['epochs']
    batch_size = config['batch_size']
    latent_dim = config['latent_dim']
    learning_rate = config['learning_rate']
    save_dir = config['save_dir']
    model_name = config['model_name']
    pretrained_model = config.get('pretrained_model', None)
    
    # Crear directorios si no existen
    os.makedirs(save_dir, exist_ok=True)
    
    # Inicializar el modelo
    print("Inicializando modelo...")
    
    # Crear expertos
    math_expert = MathExpert(input_dim=latent_dim)
    language_expert = LanguageExpert(input_dim=latent_dim, vocab_size=1000)
    
    experts = {
        'math': math_expert,
        'language': language_expert
    }
    
    # Crear componentes
    router = VeloraRouter(input_dim=latent_dim)
    latent_space = SharedLatentSpace(latent_dim=latent_dim)
    fusion = OutputFusion(input_dim=latent_dim, num_experts=len(experts))
    
    # Crear modelo completo
    model = VeloraModel(
        latent_dim=latent_dim,
        experts=experts,
        router=router,
        latent_space=latent_space,
        fusion=fusion
    )
    
    # Cargar modelo preentrenado si se especifica
    if pretrained_model and os.path.exists(pretrained_model):
        print(f"Cargando modelo preentrenado: {pretrained_model}")
        model.load_state_dict(torch.load(pretrained_model))
    
    print("Modelo inicializado correctamente")
    
    # Verificar si los archivos de datos existen
    if not os.path.exists(train_file):
        print(f"ERROR: Archivo de entrenamiento no encontrado: {train_file}")
        return False
    
    if not os.path.exists(val_file):
        print(f"ERROR: Archivo de validación no encontrado: {val_file}")
        return False
    
    # Preparar dataloaders
    print("Preparando datos...")
    train_loader, val_loader = create_data_loaders(
        train_file,
        val_file,
        batch_size=batch_size,
        input_dim=latent_dim
    )
    
    print(f"DataLoaders inicializados: {len(train_loader.dataset)} muestras de entrenamiento, {len(val_loader.dataset)} de validación")
    
    # Definir optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()
    
    # Guardar historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'domain_accuracy': [],
        'val_domain_accuracy': []
    }
    
    # Inicializar variables para seguimiento
    best_val_loss = float('inf')
    start_time = time.time()
    
    # Entrenamiento
    print("\nComenzando entrenamiento...")
    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Entrenamiento
            model.train()
            train_loss = 0.0
            correct_domain = 0
            total_samples = 0
            
            for batch_idx, (inputs, metadata) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Determinar si es secuencial o plano
                is_sequential = inputs.dim() > 2
                input_type = 'text' if is_sequential else 'numbers'
                
                # Forward pass
                outputs, output_metadata = model(inputs, input_type=input_type)
                
                # Calcular pérdida
                # Usamos el resultado esperado del metadata como target
                target = metadata['result']
                
                # Asegurar dimensiones compatibles
                if outputs.size() != target.size():
                    # Ajustar dimensiones para comparar correctamente
                    if outputs.dim() < target.dim():
                        outputs = outputs.unsqueeze(1)
                    elif target.dim() < outputs.dim():
                        target = target.unsqueeze(1)
                
                loss = criterion(outputs, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Estadísticas
                train_loss += loss.item()
                
                # Calcular precisión del clasificador de dominio
                predicted_domain = output_metadata['routing']['primary_domain']
                true_domain = metadata['domain']
                
                correct_domain += (predicted_domain == true_domain).sum().item()
                total_samples += predicted_domain.size(0)
                
                # Mostrar progreso
                if (batch_idx + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"Epoch: {epoch}/{epochs}, Batch: {batch_idx+1}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Tiempo: {elapsed:.2f}s")
            
            # Validación
            model.eval()
            val_loss = 0.0
            val_correct_domain = 0
            val_total_samples = 0
            
            with torch.no_grad():
                for inputs, metadata in val_loader:
                    # Determinar si es secuencial o plano
                    is_sequential = inputs.dim() > 2
                    input_type = 'text' if is_sequential else 'numbers'
                    
                    # Forward pass
                    outputs, output_metadata = model(inputs, input_type=input_type)
                    
                    # Calcular pérdida
                    target = metadata['result']
                    
                    # Asegurar dimensiones compatibles
                    if outputs.size() != target.size():
                        if outputs.dim() < target.dim():
                            outputs = outputs.unsqueeze(1)
                        elif target.dim() < outputs.dim():
                            target = target.unsqueeze(1)
                    
                    loss = criterion(outputs, target)
                    val_loss += loss.item()
                    
                    # Calcular precisión del clasificador de dominio
                    predicted_domain = output_metadata['routing']['primary_domain']
                    true_domain = metadata['domain']
                    
                    val_correct_domain += (predicted_domain == true_domain).sum().item()
                    val_total_samples += predicted_domain.size(0)
            
            # Calcular estadísticas promedio
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            domain_accuracy = correct_domain / total_samples if total_samples > 0 else 0
            val_domain_accuracy = val_correct_domain / val_total_samples if val_total_samples > 0 else 0
            
            # Guardar en historial
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['domain_accuracy'].append(domain_accuracy)
            history['val_domain_accuracy'].append(val_domain_accuracy)
            
            # Actualizar scheduler
            scheduler.step(val_loss)
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{save_dir}/{model_name}_best.pt")
                print(f"Checkpoint guardado: mejor modelo ({model_name}_best.pt)")
            
            # Guardar checkpoint periódico
            if epoch % 10 == 0 or epoch == epochs:
                torch.save(model.state_dict(), f"{save_dir}/{model_name}_epoch_{epoch}.pt")
                print(f"Checkpoint guardado: {model_name}_epoch_{epoch}.pt")
            
            # Calcular tiempo
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Imprimir estadísticas
            print(f"\nEpoch: {epoch}/{epochs} - Tiempo: {epoch_time:.2f}s (Total: {total_time:.2f}s)")
            print(f"  Train Loss: {train_loss:.4f}, Domain Accuracy: {domain_accuracy:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Domain Accuracy: {val_domain_accuracy:.4f}")
            
            # Guardar historial de entrenamiento
            with open(f"{save_dir}/{model_name}_history.json", 'w') as f:
                json.dump(history, f, indent=4)
    
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido por el usuario")
    
    # Guardar modelo final
    final_path = f"{save_dir}/{model_name}_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nModelo final guardado en '{final_path}'")
    
    # Guardar historial final
    with open(f"{save_dir}/{model_name}_history.json", 'w') as f:
        json.dump(history, f, indent=4)
    
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time:.2f} segundos.")
    return True

# Funciones para pruebas rápidas del modelo
def test_model(model_path, test_file, batch_size=32, latent_dim=128):
    """Prueba un modelo en un conjunto de datos."""
    print("\n" + "="*50)
    print(f"Evaluando modelo: {model_path}")
    print("="*50)
    
    if not os.path.exists(model_path):
        print(f"ERROR: Modelo no encontrado: {model_path}")
        return
    
    if not os.path.exists(test_file):
        print(f"ERROR: Archivo de prueba no encontrado: {test_file}")
        return
    
    # Inicializar modelo
    math_expert = MathExpert(input_dim=latent_dim)
    language_expert = LanguageExpert(input_dim=latent_dim, vocab_size=1000)
    
    experts = {
        'math': math_expert,
        'language': language_expert
    }
    
    router = VeloraRouter(input_dim=latent_dim)
    latent_space = SharedLatentSpace(latent_dim=latent_dim)
    fusion = OutputFusion(input_dim=latent_dim, num_experts=len(experts))
    
    model = VeloraModel(
        latent_dim=latent_dim,
        experts=experts,
        router=router,
        latent_space=latent_space,
        fusion=fusion
    )
    
    # Cargar pesos
    try:
        model.load_state_dict(torch.load(model_path))
        print("Modelo cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return
    
    # Preparar dataset de prueba
    test_dataset = VeloraCustomDataset(test_file, input_dim=latent_dim)
    
    # Función de collate
    def collate_fn(batch):
        inputs, metadatas = zip(*batch)
        
        # Verificar si los inputs son secuenciales o planos
        is_sequential = any(input.dim() > 1 for input in inputs)
        
        if is_sequential:
            # Procesar entradas secuenciales (dominio de lenguaje)
            max_len = max(input.size(0) for input in inputs)
            batch_size = len(inputs)
            
            # Crear tensor con padding
            batch_inputs = torch.zeros(batch_size, max_len, latent_dim)
            
            for i, input in enumerate(inputs):
                seq_len = input.size(0)
                batch_inputs[i, :seq_len, :] = input
            
        else:
            # Procesar entradas planas (dominio matemático)
            batch_inputs = torch.stack(inputs)
        
        # Procesar metadatas
        batch_metadata = {}
        keys = metadatas[0].keys()
        
        for key in keys:
            if key == 'result':
                # Los resultados pueden tener diferentes formas
                batch_metadata[key] = torch.stack([m[key] for m in metadatas])
            else:
                # Otros metadatos son más simples
                batch_metadata[key] = torch.cat([m[key] for m in metadatas])
        
        return batch_inputs, batch_metadata
    
    # Crear dataloader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Evaluar modelo
    model.eval()
    criterion = nn.MSELoss()
    test_loss = 0.0
    correct_domain = 0
    correct_math_op = 0
    correct_lang_task = 0
    total_samples = 0
    total_math = 0
    total_lang = 0
    
    with torch.no_grad():
        for inputs, metadata in test_loader:
            # Determinar si es secuencial o plano
            is_sequential = inputs.dim() > 2
            input_type = 'text' if is_sequential else 'numbers'
            
            # Forward pass
            outputs, output_metadata = model(inputs, input_type=input_type)
            
            # Calcular pérdida
            target = metadata['result']
            
            # Asegurar dimensiones compatibles
            if outputs.size() != target.size():
                if outputs.dim() < target.dim():
                    outputs = outputs.unsqueeze(1)
                elif target.dim() < outputs.dim():
                    target = target.unsqueeze(1)
            
            loss = criterion(outputs, target)
            test_loss += loss.item()
            
            # Calcular precisión del clasificador de dominio
            predicted_domain = output_metadata['routing']['primary_domain']
            true_domain = metadata['domain']
            
            correct_domain += (predicted_domain == true_domain).sum().item()
            total_samples += predicted_domain.size(0)
            
            # Clasificar por dominio
            for i in range(len(true_domain)):
                if true_domain[i].item() == 0:  # Matemáticas
                    total_math += 1
                    if predicted_domain[i].item() == 0:
                        # Verificar operación matemática
                        pred_op = output_metadata['routing']['math_operation'][i]
                        true_op = metadata['operation'][i]
                        if pred_op == true_op:
                            correct_math_op += 1
                else:  # Lenguaje
                    total_lang += 1
                    if predicted_domain[i].item() == 1:
                        # Verificar tarea de lenguaje
                        pred_task = output_metadata['routing']['language_task'][i]
                        true_task = metadata['language_task'][i]
                        if pred_task == true_task:
                            correct_lang_task += 1
    
    # Calcular estadísticas
    test_loss /= len(test_loader)
    domain_accuracy = correct_domain / total_samples if total_samples > 0 else 0
    math_op_accuracy = correct_math_op / total_math if total_math > 0 else 0
    lang_task_accuracy = correct_lang_task / total_lang if total_lang > 0 else 0
    
    # Mostrar resultados
    print("\nResultados de la evaluación:")
    print(f"Muestras totales: {total_samples} ({total_math} matemáticas, {total_lang} lenguaje)")
    print(f"Pérdida de prueba: {test_loss:.4f}")
    print(f"Precisión de clasificación de dominio: {domain_accuracy:.4f}")
    print(f"Precisión de operación matemática: {math_op_accuracy:.4f}")
    print(f"Precisión de tarea de lenguaje: {lang_task_accuracy:.4f}")
    
    return {
        'test_loss': test_loss,
        'domain_accuracy': domain_accuracy,
        'math_op_accuracy': math_op_accuracy,
        'lang_task_accuracy': lang_task_accuracy
    }

# Función para mostrar el menú principal
def show_menu():
    print("\n" + "="*50)
    print("VELORA - Sistema de Entrenamiento")
    print("="*50)
    print("1. Entrenar nuevo modelo")
    print("2. Continuar entrenamiento (modelo existente)")
    print("3. Evaluar modelo")
    print("4. Ver modelos disponibles")
    print("5. Ver datasets disponibles")
    print("6. Configurar parámetros de entrenamiento")
    print("7. Salir")
    
    choice = input("\nSeleccione una opción (1-7): ")
    return choice

# Función para mostrar y seleccionar un modelo
def select_model():
    models = list_available_models()
    
    if not models:
        print("No hay modelos disponibles.")
        return None
    
    print("\nModelos disponibles:")
    for i, model in enumerate(models):
        info = get_model_info(os.path.join("models", model))
        print(f"{i+1}. {model} - {info}")
    
    try:
        choice = int(input("\nSeleccione un modelo (número) o 0 para cancelar: "))
        if choice == 0:
            return None
        
        if 1 <= choice <= len(models):
            return os.path.join("models", models[choice-1])
        else:
            print("Selección no válida.")
            return None
    except ValueError:
        print("Entrada no válida.")
        return None

# Función para mostrar y seleccionar un dataset
def select_dataset(prompt="Seleccione un dataset"):
    datasets = list_available_datasets()
    
    if not datasets:
        print("No hay datasets disponibles.")
        return None
    
    print("\nDatasets disponibles:")
    for i, dataset in enumerate(datasets):
        print(f"{i+1}. {dataset}")
    
    try:
        choice = int(input(f"\n{prompt} (número) o 0 para cancelar: "))
        if choice == 0:
            return None
        
        if 1 <= choice <= len(datasets):
            return os.path.join("data", datasets[choice-1])
        else:
            print("Selección no válida.")
            return None
    except ValueError:
        print("Entrada no válida.")
        return None

# Función para configurar parámetros de entrenamiento
def configure_training_params():
    # Cargar configuración actual si existe
    config = load_training_config()
    if config is None:
        # Configuración por defecto
        config = {
            'train_file': 'data/velora_train.csv',
            'val_file': 'data/velora_val.csv',
            'epochs': 100,
            'batch_size': 32,
            'latent_dim': 128,
            'learning_rate': 0.001,
            'save_dir': 'models',
            'model_name': f'velora_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
    
    print("\n" + "="*50)
    print("Configuración de Parámetros de Entrenamiento")
    print("="*50)
    print("Valores actuales:")
    for k, v in config.items():
        print(f"{k}: {v}")
    
    print("\nParametros que puede modificar:")
    print("1. Número de épocas")
    print("2. Tamaño de batch")
    print("3. Dimensión latente")
    print("4. Tasa de aprendizaje")
    print("5. Nombre del modelo")
    print("6. Cambiar todos los parámetros")
    print("7. Volver al menú principal")
    
    choice = input("\nSeleccione una opción (1-7): ")
    
    if choice == '1':
        try:
            config['epochs'] = int(input("Nuevo número de épocas: "))
        except ValueError:
            print("Valor no válido. Se mantiene el valor anterior.")
    
    elif choice == '2':
        try:
            config['batch_size'] = int(input("Nuevo tamaño de batch: "))
        except ValueError:
            print("Valor no válido. Se mantiene el valor anterior.")
    
    elif choice == '3':
        try:
            config['latent_dim'] = int(input("Nueva dimensión latente: "))
        except ValueError:
            print("Valor no válido. Se mantiene el valor anterior.")
    
    elif choice == '4':
        try:
            config['learning_rate'] = float(input("Nueva tasa de aprendizaje: "))
        except ValueError:
            print("Valor no válido. Se mantiene el valor anterior.")
    
    elif choice == '5':
        new_name = input("Nuevo nombre de modelo: ")
        if new_name:
            config['model_name'] = new_name
    
    elif choice == '6':
        try:
            config['epochs'] = int(input("Número de épocas: "))
            config['batch_size'] = int(input("Tamaño de batch: "))
            config['latent_dim'] = int(input("Dimensión latente: "))
            config['learning_rate'] = float(input("Tasa de aprendizaje: "))
            new_name = input("Nombre de modelo: ")
            if new_name:
                config['model_name'] = new_name
        except ValueError:
            print("Algún valor no es válido. Se mantienen algunos valores anteriores.")
    
    elif choice == '7':
        return config
    
    else:
        print("Opción no válida.")
    
    # Guardar configuración
    save_training_config(config)
    
    return config

# Función principal
def main():
    """Función principal que ejecuta la interfaz de menú."""
    
    # Crear directorios necesarios
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Cargar configuración inicial
    config = load_training_config()
    if config is None:
        config = {
            'train_file': 'data/velora_train.csv',
            'val_file': 'data/velora_val.csv',
            'epochs': 100,
            'batch_size': 32,
            'latent_dim': 128,
            'learning_rate': 0.001,
            'save_dir': 'models',
            'model_name': f'velora_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
        save_training_config(config)
    
    while True:
        choice = show_menu()
        
        if choice == '1':
            # Entrenar nuevo modelo
            print("\nCreando nuevo modelo...")
            
            # Seleccionar datasets
            train_file = select_dataset("Seleccione dataset de entrenamiento")
            if train_file is None:
                continue
            
            val_file = select_dataset("Seleccione dataset de validación")
            if val_file is None:
                continue
            
            # Actualizar configuración
            config['train_file'] = train_file
            config['val_file'] = val_file
            config['model_name'] = f'velora_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            config['pretrained_model'] = None
            
            # Guardar configuración
            save_training_config(config)
            
            # Iniciar entrenamiento
            train_velora(config)
        
        elif choice == '2':
            # Continuar entrenamiento
            print("\nContinuar entrenamiento de modelo existente")
            
            # Seleccionar modelo preentrenado
            pretrained_model = select_model()
            if pretrained_model is None:
                continue
            
            # Seleccionar datasets
            train_file = select_dataset("Seleccione dataset de entrenamiento")
            if train_file is None:
                continue
            
            val_file = select_dataset("Seleccione dataset de validación")
            if val_file is None:
                continue
            
            # Actualizar configuración
            config['train_file'] = train_file
            config['val_file'] = val_file
            config['pretrained_model'] = pretrained_model
            config['model_name'] = f'velora_continued_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            
            # Guardar configuración
            save_training_config(config)
            
            # Iniciar entrenamiento
            train_velora(config)
        
        elif choice == '3':
            # Evaluar modelo
            print("\nEvaluar modelo")
            
            # Seleccionar modelo
            model_path = select_model()
            if model_path is None:
                continue
            
            # Seleccionar dataset de prueba
            test_file = select_dataset("Seleccione dataset de prueba")
            if test_file is None:
                continue
            
            # Evaluar modelo
            test_model(model_path, test_file, batch_size=config['batch_size'], latent_dim=config['latent_dim'])
        
        elif choice == '4':
            # Ver modelos disponibles
            print("\nModelos disponibles:")
            models = list_available_models()
            
            if not models:
                print("No hay modelos disponibles.")
                continue
            
            for i, model in enumerate(models):
                info = get_model_info(os.path.join("models", model))
                print(f"{i+1}. {model}")
                print(f"   {info}")
                
                # Verificar si hay archivo de historial
                history_path = os.path.join("models", model.replace(".pt", "_history.json"))
                if os.path.exists(history_path):
                    with open(history_path, 'r') as f:
                        history = json.load(f)
                    if history.get('val_loss'):
                        print(f"   Mejor val_loss: {min(history['val_loss']):.4f}")
                    if history.get('val_domain_accuracy'):
                        print(f"   Mejor domain_accuracy: {max(history['val_domain_accuracy']):.4f}")
                
                print()
            
            input("\nPresione Enter para continuar...")
        
        elif choice == '5':
            # Ver datasets disponibles
            # train_velora_menu.py (continuación)
           
           # Ver datasets disponibles
           print("\nDatasets disponibles:")
           datasets = list_available_datasets()
           
           if not datasets:
               print("No hay datasets disponibles.")
               continue
           
           for i, dataset in enumerate(datasets):
               file_path = os.path.join("data", dataset)
               file_stats = os.stat(file_path)
               size_mb = file_stats.st_size / (1024 * 1024)
               mod_time = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
               
               # Contar líneas en el dataset
               line_count = 0
               try:
                   with open(file_path, 'r', encoding='utf-8') as f:
                       for _ in f:
                           line_count += 1
               except Exception:
                   line_count = "Error al contar líneas"
               
               print(f"{i+1}. {dataset}")
               print(f"   Tamaño: {size_mb:.2f} MB, Modificado: {mod_time}")
               print(f"   Muestras: {line_count-1 if isinstance(line_count, int) else line_count}")
               print()
           
           input("\nPresione Enter para continuar...")
       
        elif choice == '6':
           # Configurar parámetros de entrenamiento
           config = configure_training_params()
       
        elif choice == '7':
           # Salir
           print("\n¡Gracias por usar VELORA! Hasta pronto.")
           break
       
        else:
           print("\nOpción no válida. Por favor, intente de nuevo.")

if __name__ == "__main__":
   main()