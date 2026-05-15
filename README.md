# Principia

A lightweight automatic differentiation engine and Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN) implementation in pure Python.

## Overview

Principia explores gradient-based learning with a small from-scratch autograd system.
It uses directed acyclic graphs (DAGs) for backpropagation and composes Gaussian RBF edges into layers.

## Requirements

- Python 3.12+
- No external Python dependencies

## Quickstart

```bash
git clone https://github.com/KodingOnion/principia.git
cd principia
python -m unittest discover -s tests
python demos/demo_autograd.py
```

## Project Structure

```text
principia/
├── engine/
│   ├── layer.py      # RBF layer implementation
│   ├── rbf_edge.py   # Gaussian RBF edge implementation
│   └── value.py      # Scalar autograd Value class and backpropagation
├── demos/
│   └── demo_autograd.py
├── tests/
│   ├── test_layer.py
│   ├── test_rbf_edge.py
│   └── test_value.py
├── .github/workflows/tests.yml
└── README.md
```

## Running Tests

```bash
python -m unittest discover -s tests
```
