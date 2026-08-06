# TFM — Baseline 3 neuronal

B3 es un MLP sobre aristas que utiliza las mismas características locales y el
mismo protocolo temporal que GraphSAGE, pero sin representación de grafo ni
propagación de mensajes. Es un control neuronal más exigente que B2 y permite
atribuir a la estructura la diferencia `GraphSAGE - B3`.

## Protocolo

- `label`, `label_reason` e identidades de paquete quedan excluidos;
- `vb_in_range` se incluye como establece la memoria;
- preprocesamiento ajustado solo con train;
- BCE ponderada por el desbalance de train;
- early stopping, época y umbral elegidos solo con validación;
- test se evalúa una vez por semilla;
- diez semillas comunes con GraphSAGE;
- IC del 95 % y agregación reproducible.

## Ejecución

```bash
pixi run install-dev
pixi run test
pixi run train --device cuda
pixi run benchmark --device cuda
pixi run compare-graphsage
```

Los artefactos quedan en `results/_local/baseline3/`.

La ablación sin la señal PEP 440 es opt-in y debe escribirse en un directorio
independiente para no mezclar protocolos:

```bash
pixi run benchmark \
  --exclude-vb-in-range \
  --device cuda \
  --output-dir results/_local/baseline3-no-vb-in-range
```

Sin esa bandera, B3 mantiene `vb_in_range` como característica.

La inferencia de checkpoints existentes puede medirse sin reentrenar. El
benchmark realiza calentamientos y diez repeticiones, comprueba las salidas
contra las predicciones guardadas, sincroniza CUDA y registra automáticamente
el hardware y las versiones de software:

```bash
pixi run benchmark-inference -- \
  --benchmark-root ../tfm-artifacts/results/baseline3-v2_2 \
  --output ../tfm-artifacts/results/baseline3-v2_2/inference_benchmark.json \
  --device cuda
```

## Análisis estadístico de predicciones

La CLI de análisis consume únicamente predicciones y resúmenes existentes; no
entrena ni ajusta modelos. La configuración v2.2 compara Random Forest,
XGBoost, B3 y GraphSAGE en el protocolo principal y en la ablación:

```bash
pixi run analyze-predictions \
  --config configs/analysis-v2_2.json \
  --output-dir ../tfm-artifacts/results/analysis-v2_2 \
  --bootstrap-resamples 2000 \
  --bootstrap-seed 42 \
  --ece-bins 10
```

Antes de calcular resultados se exige el mismo hash de dataset, lista ordenada
de semillas, conjunto de `pair_id` y etiquetas en todas las entradas. Cada
experimento declara en JSON su raíz, patrón de predicciones y, opcionalmente,
los nombres de las columnas. El mismo contrato permite incorporar nuevas
variantes como experimentos y declarar comparaciones sin cambiar la
implementación.

El informe incluye métricas por semilla, `precision@10`, `precision@50`, Brier,
log-loss, ECE y tablas de calibración con intervalos equiespaciados en `[0, 1]`.
Las comparaciones usan diferencias `right - left`, bootstrap pareado por
`pair_id` con IC percentil del 95 %, Wilcoxon pareado sobre las diez semillas y
corrección de Holm sobre la familia completa de comparación por métrica.

Las salidas son `analysis.json` y tablas CSV para métricas, calibración,
comparaciones, bootstrap y contrastes Wilcoxon-Holm.

## Datos y reproducibilidad

Los experimentos finales utilizan la configuración `mvd_v2_2` de la colección
[pypi-dependency-compatibility-mvd](https://huggingface.co/datasets/keyduu/pypi-dependency-compatibility-mvd).
El repositorio de datos documenta también el piloto, MVD v1 y MVD v2.1 para
conservar la evolución metodológica. Los datos no se versionan en este
repositorio.

Para mantener la ruta predeterminada, sitúe la colección junto a este proyecto
y cree `data/_local` como enlace al directorio `data`, o pase la ruta del
Parquet mediante la opción `--dataset`.

## Licencia y cita

El código se distribuye bajo licencia [MIT](LICENSE). Los metadatos de citación
están disponibles en [CITATION.cff](CITATION.cff).
