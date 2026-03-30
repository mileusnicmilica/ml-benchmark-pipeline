# main.py
import json
import argparse
from datetime import datetime

from models import LinearNet, CNNNet, DeepNet
from data import get_dataloaders
from benchmark import (
    train_model,
    evaluate_model,
    get_model_size,
    measure_inference,
    save_results,
    generate_markdown_report,
    generate_html_report
)


def run_benchmark(epochs: int = 5):
    print("=" * 50)
    print("   ML Model Benchmark Pipeline")
    print(f"   Epochs: {epochs}")
    print("=" * 50)

    print("\n Loading MNIST dataset...")
    train_loader, test_loader = get_dataloaders(batch_size=64)

    models = {
        "LinearNet": LinearNet(),
        "CNNNet": CNNNet(),
        "DeepNet": DeepNet()
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'=' * 50}")
        print(f"  Training {name}")
        print(f"{'=' * 50}")

        loss_history, training_time = train_model(model, train_loader, epochs=epochs)
        accuracy = evaluate_model(model, test_loader)
        accuracy = evaluate_model(model, test_loader)
        inference_ms = measure_inference(model, test_loader)

        results[name] = {
            "accuracy": accuracy,
            "training_time": training_time,
            "params": get_model_size(model),
            "loss_history": loss_history,
            "inference_ms": inference_ms 
        }

    print("\n Saving results...")
    save_results(results, output_dir="results")
    generate_markdown_report(results, output_dir="results")
    generate_html_report(results, output_dir="results")
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
    parser = argparse.ArgumentParser(description="ML Model Benchmark Pipeline")
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5)"
    )
    args = parser.parse_args()
    run_benchmark(epochs=args.epochs)