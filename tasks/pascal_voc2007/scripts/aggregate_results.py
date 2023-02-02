import bocas
import numpy as np
import luketils
import pandas as pd

results = bocas.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    history = result.get("history").metrics
    mAP = np.array(history["val_AP"]).max()
    AR = np.array(history["val_ARmax100"]).max()

    cols = [result.name] + [mAP, AR]
    cols = [cols]
    colheads = ["name"] + ["mAP", "Recall"]
    all_dfs.append(pd.DataFrame(cols, columns=colheads))

df = pd.concat(all_dfs)
result = df.to_markdown()
with open("results/metrics.md", "w") as f:
    f.write(result)

metrics_to_plot = {}

for experiment in results:
    metrics = experiment.get("history").metrics
    metrics_to_plot[f"{experiment.name} MaP"] = np.array(metrics["val_AP"])
    metrics_to_plot[f"{experiment.name} Recall"] = np.array(metrics["val_ARmax100"])

luketils.visualization.line_plot(
    metrics_to_plot,
    path=f"results/map-recall-results.png",
    title="Metrics",
)
# TODO(lukewood): save as markdown table
