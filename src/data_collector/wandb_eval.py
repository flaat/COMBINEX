
import pandas as pd
import wandb

api = wandb.Api()
entity, project = "counterfactual", "COMBINEX"
runs = api.runs(entity + "/" + project)
metrics = {}
keys = [ 'Distribution Distance Projection', 'Edge Sparsity', 'Fidelity', 'GED', 'Node Sparsity', 'Time', 'Validity']
summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values
    #  for metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)
    
    
    key = " ".join(run.name.split(" ")[0].split("_")[:-1])
    if key not in metrics:
        metrics[key] = []
    metrics[key].append(list(map(run.summary._json_dict.get, keys)) )
    # .config contains the hyperparameters.
    #  We remove special values that start with _.

    # .name is the human-readable name of the run.

import numpy as np
summary = {}
for key, value in metrics.items():
    
    try:
        arr   = np.array([[float(x) if x!='NaN' else np.nan for x in row] for row in value], dtype=float)
        
        means = arr.mean(axis=0)
        stds  = arr.std(axis=0) 
    
    except:
        print(f"Error processing {key}: {value}")
        summary
    summary[key] = {
        "mean": means,
        "std": stds
    }

# Create a pandas DataFrame from the summary dictionary
df_data = []
for method, values in summary.items():
    task, explainer, dataset, model, _ = method.split(" ")
    row = {"Task": task, "Explainer": explainer, "Dataset": dataset, "Model": model}
    for i, key in enumerate(keys):
        row[f'{key} Mean'] = values['mean'][i]
        row[f'{key} Std'] = values['std'][i]
    df_data.append(row)

summary_df = pd.DataFrame(df_data)

# Print the DataFrame
print(summary_df)

# Optionally save to CSV
summary_df.to_csv("summary_metrics.csv", index=False)

