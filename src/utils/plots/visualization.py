from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pickle
from sklearn.svm import SVC
import torch
import os
import numpy as np
import random
import imageio
import matplotlib.pyplot as plt
from torch_geometric.utils import to_dense_adj
from src.node_level_explainer.utils.utils import normalize_adj

np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)



def get_data_distribution_classifier(dataset, oracle)->None:

    adj = to_dense_adj(dataset.edge_index).squeeze()  
    norm_adj = normalize_adj(adj)   
    output = oracle.get_embedding_repr(dataset.x, norm_adj)
    out = oracle(dataset.x, norm_adj)
    y_pred_orig = torch.argmax(out, dim=1)
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')  # Use 3D projection

    z = PCA(n_components=3).fit_transform(output.detach().cpu().numpy())
    ax.scatter(z[:, 0], z[:, 1], z[:, 2], s=70, alpha=1, c=y_pred_orig)
    # Fit a linear SVM for illustrative purposes
    min_values = np.min(z, axis=0)
    max_values = np.max(z, axis=0)
    ax.set_xlim(min_values[0], max_values[0])
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_zlim(min_values[2], max_values[2])
    # Create grid to evaluate model
    plt.legend()
    plt.show()


def visualize(cfg, model_output, index, idx, loss, dimensions: int = 3, path: str = None)->None:

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')  # Use 3D projection
    points = []

    for sample, c, i, label, m in zip(model_output, cfg["color"], [0, 1], cfg["labels"], cfg["markers"]):
        # Adjust n_components to 3 for 3D
        z = PCA(n_components=3).fit_transform(sample.detach().cpu().numpy())
        colors = [c]*(sample.shape[0]-1)
        points.append(z)
        mask = torch.arange(sample.shape[0]) != index
        if dimensions == 3:
            ax.scatter(z[mask, 0], z[mask, 1], z[mask, 2], s=70, c=colors, label=label)
            ax.scatter(z[index, 0], z[index, 1], z[index, 2], s=100, c=cfg["color"][i], marker=m)

    for start, end in zip(points[0], points[1]):
        # Line from start to end
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], cfg["arrow_color"])
        # Calculate the direction of the arrow
        direction = end - start
        length = np.linalg.norm(direction)
        direction = direction / length
        # Create arrowhead as a small line segment (adjust size as needed)
        arrow_length_ratio = 0.1
        arrow_head_length = length * arrow_length_ratio
        arrow_head_end = end - direction * arrow_head_length
        ax.plot([end[0], arrow_head_end[0]], [end[1], arrow_head_end[1]], [end[2], arrow_head_end[2]], cfg["arrow_color"])
        
    ax.legend(handles=cfg["legend_elements"], loc='upper right')
    min_values = np.min(z, axis=0)
    max_values = np.max(z, axis=0)

    ax.set_xlim(min_values[0], max_values[0])
    ax.set_ylim(min_values[1], max_values[1])
    ax.set_zlim(min_values[2], max_values[2])
    plt.title(f'fig_{idx} loss:{loss:.5f}')
    plt.savefig(f'{path}/fig_{idx}.png', dpi=400)
    plt.close()

def node_embedding_visualization(path: str, model, file: str):
    
    from .configs import EMBEDDING_PLOTS_3D

    with open(f'{path}/{file}/pickle/{file[:-20]}.pickle', 'rb') as input_file:
        samples = pickle.load(input_file)

    folder_path = f"./plots/embeddings/{file}"
    for num in range(len(samples)):
        if not os.path.exists(folder_path + f"/{num}") and samples[0][1] != []:
            os.makedirs(folder_path + f"/{num}")
        if samples[num][1] is None:
            continue
        norm_adj_factual = model.normalize_adj(samples[num][0].adj)
        factual_out = model.get_embedding_repr(samples[num][0].x, norm_adj_factual)
        norm_adj_counterfactual = model.normalize_adj(samples[num][1].adj)            
        counterfactual_out = model.get_embedding_repr(samples[num][1].x, norm_adj_counterfactual)
        visualize(cfg=EMBEDDING_PLOTS_3D,
                model_output=[factual_out, counterfactual_out], 
                index=samples[num][1].sub_index, 
                idx=samples[num][1].node_idx, 
                loss=sum(samples[num][1].loss.values()), 
                dimensions=3,
                path=folder_path)

def node_embedding_visualization_history(path: str, model, file: str):
    
    from .configs import EMBEDDING_PLOTS_3D

    with open(f'{path}/{file}/pickle/{file[:-20]}_HISTORY.pickle', 'rb') as input_file:
        samples = pickle.load(input_file)

    folder_path = f"./plots/embeddings/{file}"

    for num in range(len(samples)):

        if not os.path.exists(folder_path + f"/{num}") and samples[num][1] != []:
            os.makedirs(folder_path + f"/{num}")

            for i, counterfactual in enumerate(samples[num][1]):
                if counterfactual is not None:

                    norm_adj_factual = model.normalize_adj(samples[num][0].adj)
                    factual_out = model.get_embedding_repr(samples[num][0].x, norm_adj_factual)
                    norm_adj_counterfactual = model.normalize_adj(counterfactual.adj)            
                    counterfactual_out = model.get_embedding_repr(counterfactual.x, norm_adj_counterfactual)
                    visualize(cfg=EMBEDDING_PLOTS_3D,
                            model_output=[factual_out, counterfactual_out], 
                            index=counterfactual.sub_index, 
                            idx=i, 
                            loss=sum(counterfactual.loss.values()), 
                            dimensions=3,
                            path=folder_path + f"/{num}")
            make_gif(folder_path + f"/{num}")
    


def plot_losses(path: str, file: str):

    file = "TECHNIQUE:cf-gnnfeatures_DATASET:syn5_MODEL:GCNSynthetic_TASK:node_classification_SEED:42_HISTORY"
    with open(f'{path}/{file}.pickle', 'rb') as input_file:
        samples = pickle.load(input_file)

    losses = {"total": [], "feature": [], "prediction": [], "graph": []}
    for i, counterfactual in enumerate(samples[2][1]):
        if counterfactual is not None:

            f = counterfactual.loss["feature"]
            p = counterfactual.loss["prediction"]
            g = counterfactual.loss["graph"]

            losses["total"].append(f+g+p)
            losses["feature"].append(f)
            losses["prediction"].append(p)
            losses["graph"].append(g)
    
    plot_losses_separate_axes(losses)


def plot_losses_separate_axes(losses):

    from .configs import LOSSES_PLOTS

    # Create a figure and a set of subplots
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
    for ax, name, color in zip(axs, LOSSES_PLOTS["names"],LOSSES_PLOTS["colors"]):

        # Total Loss
        ax.plot(losses[name], label=f'{name} loss',  color=color)
        ax.set_title(f'{name} loss')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.legend()

    # Adjust layout to make room for the plots
    plt.tight_layout()
    plt.savefig(f'losses.png', dpi=400)


def make_gif(path: str):

    figures = len(os.listdir(path))


    # Filepaths for the frames
    filenames = [f'fig_{i}.png' for i in range(figures)]

    # Create a GIF
    with imageio.get_writer(f'{path}/cf.gif', mode='I', duration=0.45) as writer:  # Adjust duration as needed
        for filename in filenames:
            image = imageio.imread(path+"/"+filename)
            writer.append_data(image)