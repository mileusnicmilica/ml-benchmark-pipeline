# benchmark/reporter.py
import json
import os
from datetime import datetime


def save_results(results: dict, output_dir: str = "results"):
    """
    Saves benchmark results to JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"benchmark_{timestamp}.json")

    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


def generate_markdown_report(results: dict, output_dir: str = "results"):
    """
    Generates a Markdown report from benchmark results.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "latest_report.md")

    with open(filepath, "w") as f:
        f.write("# ML Model Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Results\n\n")
        f.write("| Model | Accuracy | Training Time | Parameters |\n")
        f.write("|-------|----------|---------------|------------|\n")

        for model_name, data in results.items():
            f.write(
                f"| {model_name} "
                f"| {data['accuracy']:.2f}% "
                f"| {data['training_time']:.1f}s "
                f"| {data['params']:,} |\n"
            )

        f.write("\n## Winner\n\n")
        best = max(results, key=lambda x: results[x]["accuracy"])
        f.write(f" **{best}** with **{results[best]['accuracy']:.2f}%** accuracy\n")

    print(f"Markdown report saved to: {filepath}")
    return filepath