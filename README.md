# 🧠 MyDiffMLP – Minimal Neural Network + AutoDiff Framework in Julia

**MyDiffMLP** is a lightweight educational deep learning framework written in Julia, built from scratch. It includes:

- a custom **reverse-mode automatic differentiation engine** (`MyAD`)
- a modular **neural network layer system** (`MyNN`)
- training support with **Adam optimization**
- an NLP preprocessing pipeline for **IMDB sentiment classification**
- demos on **toy datasets** like the **two spirals**

---

## 📦 Features

- ✅ Reverse-mode autodiff via computation graph
- ✅ Custom `Dense`, `Dropout`, and `Chain` layers
- ✅ Binary classification with cross-entropy loss
- ✅ `Adam` optimizer
- ✅ Text preprocessing and TF-IDF vectorization using `TextAnalysis.jl`
- ✅ Training and evaluation scripts for both **tabular** and **text** data

---

## 📁 Project Structure

```
MyDiffMLP/
├── src/
│   ├── MyAD.jl            # autodiff engine
│   ├── MyNN.jl            # neural network layers and training utils
│   ├── data_prep.jl       # IMDB text preprocessing pipeline
│   └── MyDiffMLP.jl       # main module
├── notebooks/
│   └── IMDB_Training_Notebook.ipynb
├── data/
│   └── imdb_dataset.jld2  # (ignored in Git)
├── Project.toml
├── README.md
```

---

## 🚀 Getting Started

### 1. Clone and activate the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 2. Prepare the dataset (optional if saved already):

```julia
include("src/data_prep.jl")
data = prepare_dataset(10000, 0.8)
```

### 3. Train the model:

Use either:
- [`notebooks/IMDB_Training_Notebook.ipynb`](notebooks/IMDB_Training_Notebook.ipynb)
- or your own script

---

## 📊 Sample output (binary classification)

```
Epoch: 1    Train Loss: 0.67    Train Acc: 0.60    Test Acc: 0.58
Epoch: 200  Train Loss: 0.04    Train Acc: 0.99    Test Acc: 0.86
...
```

---

## 📚 Dependencies

- Julia `1.9+`
- `JLD2.jl`, `TextAnalysis.jl`, `Flux.jl`, `FileIO.jl`, `IJulia`, `Statistics`

Install with:

```julia
Pkg.instantiate()
```

---
