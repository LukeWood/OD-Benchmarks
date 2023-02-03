import bocas
import luketils
import numpy as np
import pandas as pd

results = bocas.Result.load_collection("artifacts/")

all_dfs = []
for result in results:
    config = result.config
    history = result.get("history").metrics
    mAP = np.array(history["val_AP"]).max()
    AR = np.array(history["val_ARmax100"]).max()

    cols = []
    colheads = []

    prts = config.backbone.split("-")

    cols += [prts[0]]
    colheads += ["backbone"]

    cols += [prts[1]]
    colheads += ["weights"]

    cols += [config.augmenter]
    colheads += ["augmenter"]

    cols += [np.array(history["loss"]).max()]
    colheads += ["loss"]

    cols += [mAP, AR]
    colheads += ["mAP", "Recall"]

    cols = [cols]
    all_dfs.append(pd.DataFrame(cols, columns=colheads))

df = pd.concat(all_dfs)
result = df.to_markdown()
with open("results/metrics.md", "w") as f:
    f.write("# PascalVOC Results\n")
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
