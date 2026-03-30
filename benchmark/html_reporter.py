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
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Napravi tabelu redove
    table_rows = ""
    for m in names:
        css_class = "best" if m == best else ""
        acc = round(results[m]["accuracy"], 2)
        t = round(results[m]["training_time"], 1)
        inf = round(results[m]["inference_ms"], 3)
        p = "{:,}".format(results[m]["params"])
        table_rows += f'<tr><td class="{css_class}">{m}</td><td class="{css_class}">{acc}%</td><td>{t}s</td><td>{inf}ms</td><td>{p}</td></tr>\n'

    # Napravi loss dataset
    colors = ['#4C72B0', '#DD8452', '#55A868']
    loss_datasets = []
    for i, m in enumerate(names):
        loss_data = str(results[m]["loss_history"])
        loss_datasets.append(
            "{label: '" + m + "', data: " + loss_data +
            ", borderColor: '" + colors[i] + "', tension: 0.3, fill: false}"
        )
    loss_datasets_str = "[" + ", ".join(loss_datasets) + "]"
    epochs_list = str(list(range(1, len(list(results.values())[0]["loss_history"]) + 1)))

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML Benchmark Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; background: #0f1117; color: #ffffff; margin: 0; padding: 20px; }
        h1 { text-align: center; color: #4C72B0; }
        .subtitle { text-align: center; color: #888; margin-bottom: 30px; }
        .winner { text-align: center; background: #1e2130; padding: 15px; border-radius: 10px; margin-bottom: 30px; font-size: 1.2em; }
        .winner span { color: #ffd700; font-weight: bold; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
        .card { background: #1e2130; border-radius: 10px; padding: 20px; }
        .card h3 { color: #4C72B0; margin-top: 0; }
        table { width: 100%; border-collapse: collapse; background: #1e2130; border-radius: 10px; overflow: hidden; }
        th { background: #4C72B0; padding: 12px; text-align: left; }
        td { padding: 12px; border-bottom: 1px solid #2e3250; }
        tr:last-child td { border-bottom: none; }
        .best { color: #ffd700; font-weight: bold; }
    </style>
</head>
<body>
    <h1>ML Model Benchmark Dashboard</h1>
    <p class="subtitle">Generated on """ + date_str + """</p>
    <div class="winner">Winner: <span>""" + best + """</span> with <span>""" + str(round(results[best]["accuracy"], 2)) + """%</span> accuracy</div>
    <div class="grid">
        <div class="card"><h3>Accuracy (%)</h3><canvas id="accuracyChart"></canvas></div>
        <div class="card"><h3>Training Time (seconds)</h3><canvas id="timeChart"></canvas></div>
        <div class="card"><h3>Inference Time (ms)</h3><canvas id="inferenceChart"></canvas></div>
        <div class="card"><h3>Training Loss per Epoch</h3><canvas id="lossChart"></canvas></div>
    </div>
    <table>
        <tr><th>Model</th><th>Accuracy</th><th>Training Time</th><th>Inference Time</th><th>Parameters</th></tr>
        """ + table_rows + """
    </table>
    <script>
        const colors = ['#4C72B0', '#DD8452', '#55A868'];
        const names = """ + str(names) + """;
        new Chart(document.getElementById('accuracyChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: """ + str(accuracies) + """, backgroundColor: colors }] },
            options: { plugins: { legend: { display: false } }, scales: { y: { min: 90 } } }
        });
        new Chart(document.getElementById('timeChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: """ + str(times) + """, backgroundColor: colors }] },
            options: { plugins: { legend: { display: false } } }
        });
        new Chart(document.getElementById('inferenceChart'), {
            type: 'bar',
            data: { labels: names, datasets: [{ data: """ + str(inference) + """, backgroundColor: colors }] },
            options: { plugins: { legend: { display: false } } }
        });
        new Chart(document.getElementById('lossChart'), {
            type: 'line',
            data: { labels: """ + epochs_list + """, datasets: """ + loss_datasets_str + """ },
            options: { plugins: { legend: { labels: { color: '#fff' } } } }
        });
    </script>
</body>
</html>"""

    with open(filepath, "w") as f:
        f.write(html)

    print(f"HTML dashboard saved to: {filepath}")
    return filepath