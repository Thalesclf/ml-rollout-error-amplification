# Experiment environment

The following versions were reported from the environment used for this work:

- Python: **3.13.14**
- PyTorch: **2.14.0+cpu**
- PyTorch backend for the reported environment: **CPU**

The exact NumPy and Matplotlib versions were not supplied and are therefore not claimed here.

To record a complete environment snapshot on the original machine, run:

```bash
python --version
python -c "import torch; print(torch.__version__)"
python -c "import numpy; print(numpy.__version__)"
python -c "import matplotlib; print(matplotlib.__version__)"
pip freeze > requirements-lock.txt
```
