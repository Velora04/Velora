# VELORA

## Aprendizaje Experto Versátil para Razonamiento y Análisis Operacional

VELORA es un sistema avanzado de inteligencia artificial que implementa una arquitectura de Mezcla de Expertos (MoE) jerárquica inspirada en la especialización modular del cerebro humano. El sistema está diseñado para manejar múltiples dominios de razonamiento a través de módulos expertos especializados que son coordinados por sistemas inteligentes de enrutamiento neural.

![Arquitectura VELORA](docs/images/velora_architecture.png)

## Características principales

- **Arquitectura MoE Jerárquica**: Sistema modular con expertos especializados por dominio, permitiendo profundidad en capacidades específicas mientras mantiene flexibilidad general.

- **Enrutamiento Neural Dinámico**: Sistema inteligente que determina el experto o combinación de expertos más adecuados para cada consulta mediante análisis contextual profundo.

- **Fundamentos Compartidos**: Tokenizador unificado y representaciones comunes que facilitan la comunicación efectiva entre módulos especializados.

- **Memoria de Trabajo Explícita**: Sistema que mantiene contexto a través de operaciones, facilitando el razonamiento multietapa y la coherencia en interacciones extendidas.

- **Capacidades de Dominios Múltiples**: Experticia especializada en razonamiento aritmético y procesamiento de lenguaje natural, expandible a nuevos dominios.

## Arquitectura del sistema

VELORA organiza sus componentes en una estructura de capas jerárquicas:

### Capa de Fundamentos
- **Tokenizador Unificado**: Convierte texto y expresiones numéricas en tokens procesables
- **Marco de Embedding**: Transforma tokens en representaciones vectoriales ricas
- **Normalización de Representaciones**: Estabiliza y estandariza información entre componentes

### Capa de Coordinación
- **Enrutador Neural Primario**: Determina dominio y dirige consultas a expertos apropiados
- **Sistema de Memoria de Trabajo**: Mantiene información contextual entre operaciones
- **Gestor de Contexto**: Administra diferentes niveles de persistencia de información

### Capa de Experticia
- **MoE Aritmético**: Especializado en operaciones numéricas con expertos por operación
- **MoE de Lenguaje**: Especializado en procesamiento lingüístico con expertos por tipo de consulta

### Capa de Integración
- **Alineación de Representaciones**: Armoniza salidas de diferentes dominios
- **Verificación de Consistencia**: Evalúa coherencia entre resultados de expertos
- **Resolución de Conflictos**: Maneja desacuerdos entre expertos mediante estrategias adaptativas

## Instalación

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/velora.git
cd velora

# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar en modo desarrollo
pip install -e .
```

## Requisitos

- Python 3.8 o superior
- PyTorch 2.0 o superior
- CUDA recomendado para entrenamiento

## Guía rápida

### Entrenamiento de componentes

```bash
# Entrenar tokenizador
python scripts/training/train_tokenizer.py --data_dir data/raw --output_dir data/processed

# Entrenar experto aritmético
python scripts/training/train_math_expert.py --data_file data/processed/math_dataset.csv --output_dir checkpoints/math_expert

# Entrenar experto de lenguaje
python scripts/training/train_language_expert.py --data_file data/processed/language_dataset.csv --output_dir checkpoints/language_expert
```

### Entrenamiento del sistema completo

```bash
# Entrenar VELORA completo usando expertos pre-entrenados
python scripts/training/train_full_model.py \
    --math_expert checkpoints/math_expert/best_model.pt \
    --language_expert checkpoints/language_expert/best_model.pt \
    --data_dir data/processed \
    --output_dir checkpoints/velora
```

### Evaluación

```bash
# Evaluar modelo entrenado
python scripts/evaluation/evaluate_model.py --model_path checkpoints/velora/velora_final.pt --test_file data/processed/test_dataset.csv
```

### Demostración interactiva

```bash
# Ejecutar demo interactiva
python scripts/demo.py --model_path checkpoints/velora/velora_final.pt --interactive
```

## Estructura del proyecto

```
velora/
├── config/                # Configuraciones del modelo
│   ├── model_configs/     # Configuraciones por componente
│   ├── training_configs/  # Configuraciones de entrenamiento
│   └── evaluation_configs/ # Configuraciones de evaluación
├── data/                  # Datos para entrenamiento y evaluación
│   ├── raw/               # Datos sin procesar
│   └── processed/         # Datos procesados
├── src/                   # Código fuente principal
│   ├── data/              # Procesamiento y gestión de datos
│   │   ├── datasets/      # Implementaciones de datasets
│   │   ├── tokenization/  # Tokenizador y procesamiento de texto
│   │   └── processing/    # Funciones de preprocesamiento
│   ├── models/            # Componentes del modelo
│   │   ├── components/    # Componentes fundamentales compartidos
│   │   ├── routers/       # Implementaciones de enrutadores
│   │   ├── experts/       # Expertos especializados
│   │   │   ├── arithmetic/ # Expertos matemáticos
│   │   │   └── language/  # Expertos lingüísticos
│   │   ├── integration/   # Componentes de integración
│   │   └── velora.py      # Modelo principal
│   ├── training/          # Lógica de entrenamiento
│   │   ├── objectives/    # Funciones de pérdida
│   │   ├── optimizers/    # Optimizadores personalizados
│   │   ├── schedulers/    # Schedulers para tasa de aprendizaje
│   │   └── trainers/      # Implementaciones de entrenadores
│   └── evaluation/        # Herramientas de evaluación
│       ├── metrics/       # Métricas de evaluación
│       ├── analysis/      # Herramientas de análisis
│       └── visualization/ # Visualización de resultados
├── scripts/               # Scripts ejecutables
│   ├── data_preparation/  # Preparación de datos
│   ├── training/          # Scripts de entrenamiento
│   ├── evaluation/        # Scripts de evaluación
│   └── deployment/        # Scripts para despliegue
├── notebooks/             # Notebooks para análisis y demos
├── tests/                 # Tests unitarios e integración
├── docs/                  # Documentación
│   ├── images/            # Imágenes y diagramas
│   └── api/               # Documentación de API
├── checkpoints/           # Modelos entrenados (guardados localmente)
├── LICENSE                # Licencia del proyecto
├── README.md              # Este archivo
├── setup.py               # Configuración de instalación
└── requirements.txt       # Dependencias del proyecto
## Metodología de entrenamiento

VELORA utiliza un enfoque de entrenamiento progresivo en múltiples fases:

1. **Entrenamiento de componentes fundamentales**: El tokenizador y el marco de embedding se entrenan primero con un corpus diverso que abarca contenido tanto aritmético como lingüístico.

2. **Entrenamiento de expertos especializados**: Cada experto (aritmético, lingüístico) se entrena por separado con datos específicos de su dominio, optimizando su rendimiento en tareas concretas.

3. **Entrenamiento de enrutadores**: Los sistemas de enrutamiento se entrenan para clasificar correctamente las entradas y dirigirlas a los expertos apropiados.

4. **Integración y fine-tuning**: El sistema completo se integra y se realiza un fine-tuning con todos los componentes, con un enfoque por fases:
   - Fase 1: Solo se entrenan enrutadores y sistemas de fusión
   - Fase 2: Se descongelan parcialmente algunos expertos
   - Fase 3: Fine-tuning completo con baja tasa de aprendizaje

Este enfoque permite que cada componente desarrolle experticia especializada antes de la integración, maximizando tanto el rendimiento específico como la coherencia del sistema completo.

## Capacidades actuales

El prototipo actual de VELORA incluye:

### Dominio Aritmético
- Reconocimiento y clasificación de operaciones matemáticas
- Procesamiento preciso de las cuatro operaciones básicas
- Detección y manejo de casos especiales (división por cero, etc.)
- Verificación simbólica de resultados

### Dominio Lingüístico
- Clasificación de consultas por tipo (preguntas, comandos, declaraciones)
- Procesamiento especializado para cada tipo de consulta
- Análisis sintáctico y semántico adaptado a la tarea
- Generación de respuestas contextualmente apropiadas

### Capacidades integradas
- Enrutamiento inteligente entre dominios
- Memoria de trabajo para mantener contexto
- Explicabilidad del proceso de razonamiento
- Resolución de consultas que cruzan dominios

## Ejemplos de uso

### Consultas aritméticas
```python
from velora import VeloraProcessor

processor = VeloraProcessor.from_pretrained("checkpoints/velora/")

# Consultas aritméticas simples
result = processor.process("¿Cuánto es 25 + 18?")
print(result.value)  # 43.0

# Consultas con análisis detallado
result = processor.process("Calcula 145 - 89", explain=True)
print(result.value)  # 56.0
print(result.explanation)  # Muestra explicación del proceso
```

### Consultas lingüísticas
```python
# Procesamiento de preguntas
result = processor.process("¿Cómo funciona el enrutador neural?")
print(result.response)  # Respuesta explicativa sobre el enrutador

# Procesamiento de comandos
result = processor.process("Explica la arquitectura del sistema de memoria")
print(result.response)  # Explicación detallada sobre el sistema de memoria
```

### Modo interactivo
```python
# Iniciar sesión interactiva para mantener contexto
session = processor.create_session()

# Secuencia de consultas relacionadas
session.process("Calcula 25 × 4")  # Respuesta: 100
session.process("Ahora divide ese resultado entre 8")  # Respuesta: 12.5
session.process("¿Qué operaciones hemos realizado hasta ahora?")  # Responde con historial
```

## Extensibilidad

VELORA está diseñado para ser extensible a nuevos dominios y capacidades:

1. **Nuevos expertos**: Se pueden integrar expertos adicionales para dominios como procesamiento visual, razonamiento lógico o análisis de datos.

2. **Mejora de componentes**: Cada componente puede actualizarse independientemente, permitiendo incorporar avances técnicos específicos sin reconstruir todo el sistema.

3. **Personalización**: Los pesos de enrutamiento y prioridades pueden ajustarse para diferentes casos de uso, equilibrando precisión, velocidad y otros factores según necesidades.

## Contribuir

Las contribuciones son bienvenidas. Para contribuir:

1. Haz fork del repositorio
2. Crea una rama para tu característica (`git checkout -b feature/amazing-feature`)
3. Realiza tus cambios
4. Ejecuta las pruebas (`pytest`)
5. Haz commit de tus cambios (`git commit -am 'Add: nueva característica'`)
6. Empuja a la rama (`git push origin feature/amazing-feature`)
7. Abre un Pull Request

Por favor, asegúrate de seguir las convenciones de código y añadir pruebas para nuevas características.

## Directrices para contribuciones

- Sigue las convenciones de estilo PEP 8
- Escribe pruebas para nuevas funcionalidades
- Actualiza la documentación para reflejar cambios
- Sigue el patrón arquitectónico existente para nuevos componentes

## Equipo de desarrollo

VELORA es desarrollado y mantenido por un equipo dedicado de investigadores y desarrolladores.

## Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## Citación

Si utilizas VELORA en tu investigación, por favor cítalo como:

```
@software{velora2025,
  author = {Autor Apellido},
  title = {VELORA: Aprendizaje Experto Versátil para Razonamiento y Análisis Operacional},
  year = {2025},
  url = {https://github.com/tu-usuario/velora}
}
```

## Agradecimientos

- A los investigadores cuyos trabajos en mezcla de expertos y arquitecturas neuronales han inspirado este proyecto
- A las comunidades de código abierto de PyTorch y Hugging Face por sus herramientas excepcionales
- A todos los contribuidores que han ayudado a mejorar VELORA

## Contacto

Para preguntas o sugerencias, por favor abre un issue en este repositorio o contacta al equipo en velora@ejemplo.com