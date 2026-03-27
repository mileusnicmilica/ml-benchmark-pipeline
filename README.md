# ML Model Benchmark Pipeline

A CI/CD pipeline that automatically trains, evaluates, and compares PyTorch models on the MNIST dataset — containerized with Docker and automated with GitHub Actions.

## Overview

This project benchmarks three neural network architectures:
- **LinearNet** — simple fully-connected baseline
- **CNNNet** — convolutional network for spatial pattern recognition  
- **DeepNet** — deeper fully-connected network with BatchNorm

Each model is evaluated on accuracy, training time, and parameter count. Results are automatically saved as JSON and Markdown reports.

## Results

| Model | Accuracy | Training Time | Parameters |
|-------|----------|---------------|------------|
| LinearNet | 97.48% | 102.0s | 109,386 |
| CNNNet | 99.17% | 212.2s | 421,642 |
| DeepNet | 98.07% | 129.7s | 576,586 |

**Winner: CNNNet with 99.17% accuracy**

## Project Structure
```
ml-benchmark-pipeline/
├── .github/workflows/    # CI/CD pipeline
├── models/               # PyTorch model architectures
│   ├── linear_net.py
│   ├── cnn_net.py
│   └── deep_net.py
├── benchmark/            # Training, evaluation, reporting
│   ├── runner.py
│   └── reporter.py
├── data/                 # MNIST data loader
├── results/              # Auto-generated reports
├── Dockerfile
├── docker-compose.yml
└── main.py
```

## How to Run

### Locally
```bash
pip install -r requirements.txt
python main.py
```

### With Docker
```bash
docker compose up --build
```

## Tech Stack

- **PyTorch** — model training and evaluation
- **Docker** — containerization for reproducibility
- **GitHub Actions** — automated CI/CD pipeline
- **Python** — scripting, automation, reporting