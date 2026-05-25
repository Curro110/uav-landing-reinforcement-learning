# Control de Aterrizaje Autónomo de UAVs mediante DRL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Stable Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-purple.svg)](https://stable-baselines3.readthedocs.io/)

Este repositorio contiene el código fuente para el entrenamiento e inferencia de un agente de Inteligencia Artificial capaz de controlar un vehículo aéreo no tripulado (cuadricóptero) para realizar aterrizajes de precisión bajo condiciones probabilísticas severas.

El proyecto prescinde de controladores PID tradicionales, utilizando Deep Reinforcement Learning (DRL) —específicamente el algoritmo Soft Actor-Critic (SAC)— para mapear estados inerciales directamente a comandos de empuje en los motores.

## Características Principales

* **Física No Lineal Personalizada:** Motor físico programado desde cero con integración numérica de Runge-Kutta de 4º orden (RK4) y cinemática basada en cuaterniones unitarios para evitar el bloqueo del cardán (Gimbal Lock). Compilado en tiempo de ejecución (JIT) vía Numba.
* **Domain Randomization:** Entrenamiento robusto frente a perturbaciones del mundo real (viento cruzado aleatorio, ruido gaussiano en actuadores e incertidumbre en la masa inercial).
* **Curriculum Learning & Transfer Learning:** El agente aprende mediante fases evolutivas, desde la estabilización básica (hovering) hasta la navegación a larga distancia (radio de 10 metros y 15 metros de altitud asimétrica).

## Estructura del Repositorio

```text
DRONE_LANDING_DRL/
├── configs/             # Coeficientes aerodinámicos y parámetros físicos
├── envs/                # Entornos de Gymnasium (parametrizados: ideal, real, extended)
├── logs/checkpoints/    # Pesos de la red neuronal entrenada (.zip y .pkl)
├── scripts/             # Orquestadores de entrenamiento e inferencia
└── utils/               # Álgebra de cuaterniones, RK4 y renderizador 3D
```
# Uso y Ejecución
La arquitectura de software está parametrizada para ser ejecutada directamente a través de argumentos por línea de comandos.

## 1. Inferencia Visual (Ejecución del modelo pre-entrenado)
Despliegue del agente operando en los distintos niveles de simulación. Puedes evaluar la red neuronal experta sometiéndola a diferentes condiciones ambientales y de distancia:
```
# 1. Mundo Ideal: Entorno de laboratorio sin viento ni variaciones inerciales
python scripts/visualize.py --model logs/checkpoints/sac_drone_extended --version ideal

# 2. Mundo Real: Inyección de viento cruzado y variaciones aleatorias de masa
python scripts/visualize.py --model logs/checkpoints/sac_drone_extended --version real

# 3. Mundo Extendido: Envolvente máxima con coordenadas asimétricas extremas
python scripts/visualize.py --model logs/checkpoints/sac_drone_extended --version extended

# Prueba de Estrés: Forzar una coordenada de aparición fuera de distribución (Ej: Z=30m)
python scripts/visualize.py --model logs/checkpoints/sac_drone_extended --version extended -
```
## 2. Entrenamiento de Políticas (Optimización SAC)
Lanzamiento de las rutinas de optimización para entrenar nuevas redes neuronales desde cero o extender las capacidades mediante aprendizaje por transferencia:
```
# Entrenar desde cero en el entorno ideal (entorno de laboratorio sin ruido)
python scripts/train.py --version ideal --timesteps 300000

# Reanudar un entrenamiento previo en el entorno extendido (Transfer Learning)
python scripts/train.py --version extended --resume --timesteps 150000
```
# Extracción de Telemetría y Gráficas Académicas
El proyecto incluye un pipeline de post-procesado que ingiere los datos .json exportados desde TensorBoard, aplicando un filtrado de Media Móvil Exponencial (EMA) para generar métricas vectoriales .pdf de calidad académica:
```
python scripts/plot_results.py
```
## Autor
Desarrollado por Francisco Delgado Capote para la asignatura de Ampliación de Robótica.
