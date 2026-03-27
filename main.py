# main.py
import json
from datetime import datetime

from models import LinearNet, CNNNet, DeepNet
from data import get_dataloaders
from benchmark import (
    train_model,
    evaluate_model,
    get_model_size,
    save_results,
    generate_markdown_report
)


def run_benchmark():
    print("=" * 50)
    print("   ML Model Benchmark Pipeline")
    print("=" * 50)

    # Učitaj podatke
    print("\n Loading MNIST dataset...")
    train_loader, test_loader = get_dataloaders(batch_size=64)

    # Modeli koje benchmarkujemo
    models = {
        "LinearNet": LinearNet(),
        "CNNNet": CNNNet(),
        "DeepNet": DeepNet()
    }

    results = {}

    # Treniraj i evaluiraj svaki model
    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"  Training {name}")
        print(f"{'=' * 50}")

        loss_history, training_time = train_model(model, train_loader, epochs=5)
        accuracy = evaluate_model(model, test_loader)

        results[name] = {
            "accuracy": accuracy,
            "training_time": training_time,
            "params": get_model_size(model),
            "loss_history": loss_history
        }

    # Sačuvaj rezultate
    print("\n Saving results...")
    save_results(results, output_dir="results")
    generate_markdown_report(results, output_dir="results")

    # Prikaži summary
    print("\n" + "=" * 50)
    print("   Final Benchmark Summary")
    print("=" * 50)
    print(f"{'Model':<12} {'Accuracy':>10} {'Time':>10} {'Params':>12}")
    print("-" * 48)
    for name, data in results.items():
        print(
            f"{name:<12} "
            f"{data['accuracy']:>9.2f}% "
            f"{data['training_time']:>9.1f}s "
            f"{data['params']:>12,}"
        )

    best = max(results, key=lambda x: results[x]["accuracy"])
    print(f"\n Winner: {best} with {results[best]['accuracy']:.2f}% accuracy")


if __name__ == "__main__":
    run_benchmark()