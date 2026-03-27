# benchmark/__init__.py
from .runner import train_model, evaluate_model, get_model_size
from .reporter import save_results, generate_markdown_report

__all__ = ["train_model", "evaluate_model", "get_model_size", "save_results", "generate_markdown_report"]