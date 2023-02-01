import ml_experiments
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
print(result)
# TODO(lukewood): save as markdown table
