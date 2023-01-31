import ml_collections
import pandas as pd

results = ml_collections.Result.load_collection('artifacts/')

df = pd.DataFrame()

for result in results:
    metrics = results.get('metrics')
    # TODO(lukewood): index all columns, column headers etc
    df[result.name] = metrics.as_dataframe_column()

result = df.to_markdown()
print(result)
# TODO(lukewood): save as markdown table
