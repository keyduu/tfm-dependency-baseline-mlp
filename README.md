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

## Datos y reproducibilidad

Los experimentos finales utilizan la configuración `mvd_v2_1` de la colección
[pypi-dependency-compatibility-mvd](https://huggingface.co/datasets/keyduu/pypi-dependency-compatibility-mvd).
El repositorio de datos documenta también el piloto y el MVD v1 para conservar
la evolución metodológica. Los datos no se versionan en este repositorio.

Para mantener la ruta predeterminada, sitúe la colección junto a este proyecto
y cree `data/_local` como enlace al directorio `data`, o pase la ruta del
Parquet mediante la opción `--dataset`.

## Licencia y cita

El código se distribuye bajo licencia [MIT](LICENSE). Los metadatos de citación
están disponibles en [CITATION.cff](CITATION.cff).
