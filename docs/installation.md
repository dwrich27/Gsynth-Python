# Installation

## Requirements

- Python >= 3.9
- numpy >= 1.24
- pandas >= 1.5
- matplotlib >= 3.6 (optional, required only for `plot()`)

---

## Install from GitHub

The recommended way to install the latest release is directly from GitHub:

```python
pip install git+https://github.com/dwrich27/Gsynth-Python.git
```

This installs the package and all required dependencies.

---

## Development Install

To contribute or run the tests locally, clone the repository and install in editable mode:

```python
git clone https://github.com/dwrich27/Gsynth-Python.git
cd Gsynth-Python
pip install -e ".[dev]"
```

The `[dev]` extra installs testing and documentation dependencies (pytest, mkdocs-material, etc.).

---

## Virtual Environment (Recommended)

It is good practice to install into a dedicated virtual environment:

```python
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install git+https://github.com/dwrich27/Gsynth-Python.git
```

---

## Verify the Installation

After installing, confirm the package is importable and check the version:

```python
import gsynth
print(gsynth.__version__)
```

You should see output like `0.1.0`.

To confirm the core estimator is accessible:

```python
from gsynth import gsynth as gs, plot, effect, GsynthResult
print("gsynth imported successfully")
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'gsynth'`

Make sure you installed the package in the same Python environment you are running. If you use Jupyter, verify the kernel matches the environment where you ran `pip install`.

### `ImportError` related to numpy or pandas

Check that you have compatible versions:

```python
import numpy, pandas
print(numpy.__version__, pandas.__version__)
```

numpy >= 1.24 and pandas >= 1.5 are required. Upgrade with:

```python
pip install --upgrade numpy pandas
```

### `plot()` raises `ImportError: matplotlib is required`

matplotlib is an optional dependency. Install it explicitly:

```python
pip install matplotlib
```

### SSL / certificate errors on `pip install git+https://...`

Try cloning manually first and using the development install approach, or configure your SSL certificates.

### pip version issues

Some older pip versions do not support PEP 660 editable installs. Upgrade pip first:

```python
pip install --upgrade pip
```

---

## Conda

If you manage environments with conda, install dependencies via conda-forge and then install gsynth via pip:

```python
conda create -n gsynth-env python=3.11 numpy pandas matplotlib
conda activate gsynth-env
pip install git+https://github.com/dwrich27/Gsynth-Python.git
```
