# ML Model Benchmark Pipeline

A CI/CD pipeline that automatically trains, evaluates, and compares PyTorch models on the MNIST dataset — containerized with Docker and automated with GitHub Actions.

## Overview

This project benchmarks three neural network architectures across four key metrics: **accuracy**, **training time**, **inference speed**, and **parameter count**. Results are automatically generated as a JSON file, Markdown report, and interactive HTML dashboard.

- **LinearNet** — simple fully-connected baseline, fastest inference
- **CNNNet** — convolutional network, best accuracy through spatial pattern recognition
- **DeepNet** — deeper fully-connected network with BatchNorm regularization

## How to Run

### Locally
```bash
pip install -r requirements.txt
python main.py                  # default: 5 epochs
python main.py --epochs 10      # custom epoch count
```

### With Docker
```bash
docker compose up --build
```

## Output

Each run automatically generates three output files in `results/`:
- `benchmark_<timestamp>.json` — raw results
- `latest_report.md` — Markdown summary table
- `dashboard.html` — interactive HTML dashboard with charts

## Project Structure
```
ml-benchmark-pipeline/
├── .github/workflows/      # CI/CD — runs on every push
│   └── benchmark.yml
├── models/                 # PyTorch model architectures
│   ├── linear_net.py
│   ├── cnn_net.py
│   └── deep_net.py
├── benchmark/              # Core pipeline logic
│   ├── runner.py           # Training, evaluation, inference timing
│   ├── reporter.py         # JSON + Markdown report generation
│   └── html_reporter.py    # Interactive HTML dashboard
├── data/                   # MNIST data loader
├── results/                # Auto-generated reports (gitignored)
├── Dockerfile
├── docker-compose.yml
└── main.py                 # Pipeline entrypoint
```

## Tech Stack

- **PyTorch** — model training and evaluation
- **Docker** — containerization for reproducibility
- **GitHub Actions** — automated CI/CD pipeline
- **Python** — scripting, automation, reporting
- **Chart.js** — interactive benchmark visualizations