# Principia

A lightweight Automatic Differentiation engine and Radial Basis Function Kolmogorov-Arnold Network (RBF-KAN) built from mathematical first principles in Python.

## Overview

Unlike standard Multi-Layer Perceptrons (MLPs) that rely on fixed activation functions at the nodes, Kolmogorov-Arnold Networks place learnable functions on the edges. **Principia** implements a custom Autograd engine to power an RBF-KAN—a computationally efficient variant that replaces the expensive B-splines of the original 2024 architecture with Gaussian Radial Basis Functions (RBFs).

This project explores the intersection of continuous optimization and discrete data structures, relying heavily on directed acyclic graphs (DAGs) to compute exact algorithmic gradients via the chain rule.

## Mathematical Architecture

At the core of the engine is the computation of exact gradients without relying on symbolic manipulation or numerical approximation (which introduces floating-point errors). 

The network replaces traditional scalar weights with learnable Gaussian curves on the edges:
$$ \phi(x) = \sum_{i=1}^{N} w_i e^{-\gamma (x - \mu_i)^2} $$

During the forward pass, operations dynamically construct a computational graph. Calling `.backward()` traverses this graph in reverse topological order, applying the chain rule to update the means ($\mu$), widths ($\gamma$), and amplitudes ($w$) of the Gaussian curves.

## Technical Highlights

- **Autograd Engine:** A custom `Value` class that wraps scalar floats, dynamically tracks parent dependencies, and stores local derivative closures to construct a computational graph.
- **Abstract Data Structures:** Heavy utilization of Directed Acyclic Graphs (DAGs) for backpropagation and topological sorting algorithms.
- **Object-Oriented Design:** A strictly modular architecture separating the core calculus engine from the neural network layers and application wrapper.
- **Robustness:** Built-in data validation and error handling to manage vanishing gradients and unstable topological states.

## Project Structure

```text
principia/
├── engine/              # The core mathematical framework
│   ├── __init__.py
│   ├── value.py         # Autograd engine and topological graph nodes
│   ├── rbf_edge.py      # Gaussian mathematical connections
│   ├── layer.py         # Node/edge grouping algorithms
│   └── model.py         # High-level KAN architecture
├── app/                 # Application wrapper (In Development)
│   ├── gui.py           # Experiment dashboard and live visualization
│   └── database.py      # SQLite integration for logging experiment states
├── main.py              # Engine entry point and testing environment