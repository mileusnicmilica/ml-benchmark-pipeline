# benchmark/html_reporter.py
import os
from datetime import datetime


def generate_html_report(results: dict, output_dir: str = "results"):
    """
    Generates an HTML dashboard with benchmark results and charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "dashboard.html")

    names = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in names]
    times = [results[m]["training_time"] for m in names]
    params = [results[m]["params"] for m in names]
    inference = [results[m]["inference_ms"] for m in names]
    best = max(results, key=lambda x: results[x]["accuracy"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; background: #0f1117; color: #ffffff; margin: 0; padding: 20px; }}
        h1 {{ text-align: center; color: #4C72B0; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .winner {{ text-align: center; background: #1e2130; padding: 15px; border-radius: 10px; margin-bottom: 30px; font-size: 1.2em; }}
        .winner span {{ color: #ffd700; font-weight: bold; }}
        .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .card {{ background: #1e2130; border-radius: 10px; padding: 20px; }}
        .card h3 {{ color: #4C72B0; margin-top: 0; }}
        table {{ width: 100%; border-collapse: collapse; background: #1e2130; border-radius: 10px; overflow: hidden; }}
        th {{ background: #4C72B0; padding: 12px; text-align: left; }}
        td {{ padding: 12px; border-bottom: 1px solid #2e3250; }}
        tr:last-child td {{ border-bottom: none; }}
        .best {{ color: #ffd700; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>ML Model Benchmark Dashboard</h1>
    <p class="subtitle">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="winner">Winner: <span>{best}</span> with <span>{results[best]['accuracy']:.2f}%</span> accuracy</div>

    <div class="grid">
        <div class="card">
            <h3>Accuracy (%)</h3>
            <canvas id="accuracyChart"></canvas>
        </div>
        <div class="card">
            <h3>Training Time (seconds)</h3>
            <canvas id="timeChart"></canvas>
        </div>
        <div class="card">
            <h3>Inference Time (ms per image)</h3>
            <canvas id="inferenceChart"></canvas>
        </div>
        <div class="card">
            <h3>Training Loss per Epoch</h3>
            <canvas id="lossChart"></canvas>
        </div>
    </div>

    <table>
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Training Time</th>
            <th>Inference Time</th>
            <th>Parameters</th>
        </tr>
        {''.join(f"""
        <tr>
            <td class="{'best' if m == best else ''}">{m}</td>
            <td class="{'best' if m == best else ''}">{round(results[m]['accuracy'], 2)}%</td>
            <td>{round(results[m]['training_time'], 1)}s</td>
            <td>{round(results[m]['inference_ms'], 3)}ms</td>
            <td>{'{:,}'.format(results[m]['params'])}</td>
        </tr>""" for m in names)}
    </table>

    <script>
        const colors = ['#4C72B0', '#DD8452', '#55A868'];
        const names = {names};

        new Chart(document.getElementById('accuracyChart'), {{
            type: 'bar',
            data: {{
                labels: names,
                datasets: [{{ data: {accuracies}, backgroundColor: colors }}]
            }},
            options: {{ plugins: {{ legend: {{ display: false }} }}, scales: {{ y: {{ min: 90 }} }} }}
        }});

        new Chart(document.getElementById('timeChart'), {{
            type: 'bar',
            data: {{
                labels: names,
                datasets: [{{ data: {times}, backgroundColor: colors }}]
            }},
            options: {{ plugins: {{ legend: {{ display: false }} }} }}
        }});

        new Chart(document.getElementById('inferenceChart'), {{
            type: 'bar',
            data: {{
                labels: names,
                datasets: [{{ data: {inference}, backgroundColor: colors }}]
            }},
            options: {{ plugins: {{ legend: {{ display: false }} }} }}
        }});

        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: {list(range(1, len(list(results.values())[0]['loss_history']) + 1))},
                datasets: [{', '.join(f"{{label: '{m}', data: {results[m]['loss_history']}, borderColor: '{['#4C72B0', '#DD8452', '#55A868'][i]}', tension: 0.3, fill: false}}" for i, m in enumerate(names))}]
            }},
            options: {{ plugins: {{ legend: {{ labels: {{ color: '#fff' }} }} }} }}
        }});
    </script>
</body>
</html>"""

    with open(filepath, "w") as f:
        f.write(html)

    print(f"HTML dashboard saved to: {filepath}")
    return filepath