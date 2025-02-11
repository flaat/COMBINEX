from matplotlib.lines import Line2D


EMBEDDING_PLOTS_3D = {
    "legend_elements": [
        Line2D([0], [0], marker='o', color='w', label='Factual',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Counterfactual',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='^', color='w', label='Factual Node',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='*', color='w', label='CounterFactual Node',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='', color='black', label='Transition',
               lw=1)
    ],
    "color": ["red", "green"],
    "labels": ["factual", "counterfactual"],
    "markers": ['^', '*'],
    "arrow_color": 'gray'
}

LOSSES_PLOTS = {
    "names": ["total", "feature", "prediction", "graph"],
    "colors": ["tab:red", "tab:blue", "tab:green", "tab:orange"]
}