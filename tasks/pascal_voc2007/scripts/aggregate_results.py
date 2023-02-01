import ml_experiments
import luketils
import pandas as pd

results = ml_experiments.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    metrics = result.get("metrics")
    all_dfs.append(
        pd.DataFrame(
            [list(metrics.metrics.values()) + [result.name]],
            columns=list(metrics.metrics.keys()) + ["name"],
        )
    )

df = pd.concat(all_dfs)
result = df.to_markdown()
with open("results/metrics.md", "w") as f:
    f.write(result)

metrics_to_plot = {}

for experiment in results:
    metrics = experiment.get_artifact("history").metrics
    metrics_to_plot[f"{experiment.name} MaP"] = metrics["MaP"]
    metrics_to_plot[f"{experiment.name} Recall"] = metrics["Recall"]

luketils.visualization.line_plot(
    metrics_to_plot,
    path=f"results/combined-accuracy.png",
    title="Metrics",
)
# TODO(lukewood): save as markdown table
