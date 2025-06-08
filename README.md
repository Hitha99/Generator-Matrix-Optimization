# Generator Matrix Optimization for Error-Correcting Codes

This project focuses on optimizing generator matrices for linear error-correcting codes by minimizing the *m-height* metric. It integrates **linear programming** and **evolutionary algorithms**—including **genetic algorithms** and **simulated annealing**—to improve encoding efficiency and reliability in digital communication systems.

## 🔍 Overview

Error-correcting codes are crucial for ensuring data integrity in noisy channels. The generator matrix plays a vital role in defining these codes. By minimizing the m-height, this project aims to:

- Enhance fault tolerance
- Reduce encoding complexity
- Improve performance in high-reliability environments

## 🛠️ Techniques Used

- **Linear Programming (LP)** – for deterministic optimization of matrix properties
- **Genetic Algorithms (GA)** – for exploring global solution space
- **Simulated Annealing (SA)** – for escaping local minima and refining results
- **Custom Fitness Function** – to evaluate and guide optimization based on m-height

## 📁 Files

- `GeneratorMatrixOptimization_Error_Code_Generation.py`: Main script implementing LP, GA, and SA-based optimization pipeline.

## 🚀 How to Run

```bash
python GeneratorMatrixOptimization_Error_Code_Generation.py
