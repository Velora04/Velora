# VELORA: Versatile Language and Operations Reasoning Architecture

VELORA es un modelo de inteligencia artificial modular que combina diferentes expertos especializados para procesar tanto lenguaje natural como operaciones matemáticas, utilizando un sistema de enrutamiento neuronal avanzado.

## Arquitectura

El modelo VELORA se compone de:

1. **Enrutador Neuronal**: Determina qué experto debe procesar cada entrada.
2. **Experto Matemático**: Especializado en operaciones aritméticas.
3. **Experto en Lenguaje**: Procesa y genera texto natural.
4. **Sistema de Fusión**: Integra las salidas de los expertos.

## Características Principales

- **Modularidad**: Arquitectura extensible para añadir nuevos expertos.
- **Enrutamiento Neural**: Decisiones de enrutamiento basadas en redes neuronales.
- **Explicabilidad**: Capacidad para explicar las decisiones internas del modelo.
- **Fusión Adaptativa**: Combinación inteligente de salidas de expertos.

## Requisitos

- Python 3.8+
- PyTorch 1.8+
- NumPy
- Matplotlib (para visualizaciones)

## Instalación

```bash
# Clonar repositorio
git clone https://github.com/Velora04/Velora
cd velora

# Instalar dependencias
pip install -r requirements.txt

# Instalar el paquete en modo desarrollo
pip install -e .