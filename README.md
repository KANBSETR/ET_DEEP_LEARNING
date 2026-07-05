# Examen Final Transversal — Integración Multimodal (CNN + RNN/Transformer)

Proyecto final de la asignatura **Deep Learning (DLY0100)** — DuocUC.

Integra dos modalidades de datos (imágenes y texto) para generar una predicción combinada: se entrenan modelos independientes para cada modalidad, se extraen sus representaciones intermedias y se fusionan en un clasificador conjunto.

## Objetivo

Entrenar modelos independientes para imágenes (Fashion-MNIST, CNN) y texto (IMDB Reviews, RNN/LSTM/GRU/Transformer), y fusionar sus representaciones para una predicción combinada.

## Estructura del notebook

1. **Introducción y preprocesamiento** — carga y normalización de Fashion-MNIST.
2. **CNN — Fashion-MNIST**: arquitectura, comparación de funciones de activación, entrenamiento final, evaluación y análisis de errores.
3. **Modelos de texto — IMDB Reviews**: comparación de cuatro arquitecturas secuenciales (RNN simple, LSTM, GRU, Transformer) bajo los mismos hiperparámetros base.
4. **Integración y fusión multimodal**: extracción de embeddings de la CNN y del mejor modelo de texto, concatenación y clasificación final sobre un dataset sintético imagen-texto.
5. **Análisis comparativo y conclusiones**: comparación técnica entre las cinco arquitecturas, interpretación de métricas y propuestas de mejora.

## Resultados obtenidos

### CNN — Fashion-MNIST

| Métrica | Valor |
|---|---|
| Test accuracy | 91.48% |
| Test loss | 0.2550 |

Mejor desempeño en clases con silueta distintiva (Trouser, Sandal, Bag, Ankle boot — F1 ≥ 0.97). Menor desempeño en **Shirt** (F1 0.74), que se confunde con T-shirt/top, Pullover y Coat por la similitud de silueta a 28×28 px en escala de grises.

### Modelos de texto — IMDB Reviews

| Modelo | Accuracy | F1-score | N° parámetros |
|---|---|---|---|
| RNN simple | 0.8508 | 0.8506 | 1,296,577 |
| LSTM | 0.8708 | 0.8705 | 1,333,633 |
| GRU | 0.8507 | 0.8370 | 1,321,473 |
| **Transformer** | 0.8703 | 0.8692 | 698,817 |

Se eligió el **Transformer** para la etapa de fusión: iguala prácticamente al LSTM con un 48% menos de parámetros y permite paralelización en GPU.

### Fusión multimodal (CNN + Transformer)

| Métrica | Valor |
|---|---|
| Test accuracy | 94.75% |
| Test F1-score | 87.18% |

La etiqueta combinada sintética está desbalanceada (8,013 negativos vs. 1,987 positivos de 10,000 pares), por lo que el F1-score es la métrica más confiable para evaluar el desempeño real de la fusión.

## Datasets

- **Fashion-MNIST**: 60,000 imágenes de entrenamiento + 10,000 de test, 28×28 px en escala de grises, 10 clases de prendas de vestir. Cargado directamente desde `keras.datasets.fashion_mnist`.
- **IMDB Reviews**: 25,000 reseñas de entrenamiento / test, análisis de sentimiento binario (positivo/negativo). Cargado desde `keras.datasets.imdb` con `VOCAB_SIZE=10000` y `MAX_LEN=200`.
- **Dataset combinado (fusión)**: construido sintéticamente emparejando imágenes y reseñas 1 a 1, con una etiqueta binaria que exige usar ambas modalidades (`imagen es calzado/bolso AND reseña positiva`).

## Requisitos

```
tensorflow>=2.15
numpy
pandas
matplotlib
seaborn
scikit-learn
```
