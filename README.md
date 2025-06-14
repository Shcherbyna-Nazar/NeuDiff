
# 🧠 NeuDiff — Modular Neural Network & Automatic Differentiation in Julia

**NeuDiff** is a lightweight, educational deep learning framework written in Julia, built from scratch for transparency, extensibility, and clarity. It features:

- a custom **reverse-mode automatic differentiation engine** (`MyAD`)
- a modular **neural network system** (`MyNN`)
- full support for layers like `Dense`, `Embedding`, `Conv1D`, `MaxPool1D`, `Chain`
- optimizers including **Adam**
- utilities for text and tabular data (e.g. IMDB sentiment classification pipeline)
- reproducible demos on both **NLP** and **toy datasets** (like the two spirals)

---

## 📦 Features

- ✅ Explicit computation graph reverse-mode autodiff engine (`MyAD`)
- ✅ Modular neural network layers (`Dense`, `Embedding`, `Conv1D`, `MaxPool1D`, `Chain`, `Dropout`)
- ✅ Full training pipeline with cross-entropy loss and Adam optimizer
- ✅ Text preprocessing, tokenization, and embedding utilities
- ✅ Training and evaluation scripts for both text (IMDB) and tabular data
- ✅ Extensive unit tests vs. Flux/Zygote

---

## 📁 Project Structure

```
NeuDiff/
├── data/
│   ├── km2/
│   ├── glove_6B_50d.jld2
│   ├── imdb_dataset.jld2
│   └── imdb_dataset_prepared.jld2
├── notebooks/
│   ├── IMDB_Training_Notebook.ipynb
│   └── KM3.ipynb
├── src/
│   ├── MyAD/
│   │   ├── activations.jl      # activation functions (relu, sigmoid, etc.)
│   │   ├── exports.jl          # exported symbols for MyAD
│   │   ├── gradients.jl        # backward pass logic
│   │   ├── graph.jl            # computation graph utilities (topological sort, etc.)
│   │   ├── MyAD.jl             # main module for MyAD
│   │   ├── nodes.jl            # node type definitions (Variable, Constant, etc.)
│   │   └── ops.jl              # operator nodes (matmul, scalar ops, etc.)
│   ├── MyNN/
│   │   ├── exports.jl          # exported symbols for MyNN
│   │   ├── layers.jl           # neural network layer definitions
│   │   ├── MyNN.jl             # main module for MyNN
│   │   ├── optim.jl            # optimizers (Adam, SGD, etc.)
│   │   └── utils.jl            # utility functions (parameters, zero_gradients!, etc.)
│   ├── data_prep.jl            # IMDB dataset preprocessing pipeline
│   └── NeuDiff.jl              # main project module (exports MyAD and MyNN)
├── tests/
│   ├── performance/
│   │   ├── complex_model.jl
│   │   ├── conv1d.jl
│   │   ├── dense.jl
│   │   ├── embedding.jl
│   │   └── maxpool1d.jl
│   ├── correctness_tests.jl    # extensive unit tests vs Flux/Zygote
│   └── km3_test.jl
├── .gitignore
├── Manifest.toml
├── MLP_TFIDF_myAD_myNN.jl      # (example training script)
├── Project.toml
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone and activate the environment

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 2. Prepare the IMDB dataset (if needed)

```julia
include("src/data_prep.jl")
data = prepare_dataset(10000, 0.8)
```

### 3. Train a model

You can use:
- the sample [notebook](notebooks/IMDB_Training_Notebook.ipynb)
- or your own training script, e.g.
  ```julia
  include("src/NeuDiff.jl")
  using .NeuDiff
  # ...build and train your model...
  ```

---

## 🧪 Testing

Run correctness tests for your AD engine and NN layers:
```julia
include("tests/correctness_tests.jl")
```

---

## 📊 Sample Output (IMDB binary classification)

```
Epoch: 1     Train Loss: 0.67     Train Acc: 0.60     Test Acc: 0.58
Epoch: 5     Train Loss: 0.21     Train Acc: 0.94     Test Acc: 0.87
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

## 🧠 Why NeuDiff?

- **Educational:** Designed to reveal how neural nets and autodiff work under the hood.
- **Modular:** Add new ops/layers with ease.
- **Tested:** Unit tests check correctness vs. Flux/Zygote.

---

## ✨ Citation

If you use or build on this code, please cite [your name/project] or link to this repository.

---
