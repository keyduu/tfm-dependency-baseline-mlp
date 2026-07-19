import sys

import pandas as pd
import torch

print(f"Python: {sys.version}")
print(f"pandas: {pd.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("Entorno Baseline 3 OK")
