# COMBINEX: A Unified Counterfactual Explainer for Graph Neural Networks

This repository provides a modular and configurable framework for generating counterfactual explanations in Graph Neural Networks (GNNs) using node feature and structural perturbations.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configurations](#configurations)
    - [Dataset Options](#dataset-options)
    - [Logger Settings](#logger-settings)
    - [Model Architectures and Techniques](#model-architectures-and-techniques)
- [Examples](#examples)
- [Known Issues](#known-issues)

## Overview

COMBINEX offers a flexible setup where all configurable options are defined within YAML files in the `config` folder. This design allows users to tailor every aspect of the software—ranging from datasets, logging, and model configurations to counterfactual explanation strategies.

![Python Version](https://img.shields.io/badge/python-3.11.10-brightgreen)
![PyTorch Version](https://img.shields.io/badge/pytorch-2.5.0-brightgreen)
![PyTorch Geometric Version](https://img.shields.io/badge/torch_geometric-2.6.1-brightgreen)
![Hydra](https://img.shields.io/badge/hydracore-1.3.2-brightgreen)
![WandB](https://img.shields.io/badge/wandb-0.17.5-brightgreen)

## Installation
Create a Conda env and install **Python 3.11.10**
```setup
conda create --name combinex
conda activate combinex
conda install python=3.11.10
```
To install the required dependencies, run:

```setup
pip install -r requirements.txt
```

Ensure you have Python 3.11.10 or later and the correct versions of PyTorch, PyTorch Geometric, and Hydra installed.

## Usage

To start the software with the default Hydra configuration, simply run:

```start
python main.py
```

If you wish to override default options, read on for configuration details.

## Configurations

COMBINEX is built around a flexible configuration system managed by Hydra. All configurations are located in the `config` folder, enabling you to customize the following:

### Dataset Options

You can choose between different dataset classes:

- **planetoid**: Supports datasets such as Cora, Citeseer, Pubmed.
- **attributed**: Includes datasets like Facebook BlogCatalog and Wiki.
- **webkb**: Options include Cornell, Wisconsin, Texas.
- **karate**
- **actor**

For example, to use the Cora dataset:

```dataset
... dataset=planetoid dataset.name=cora
```

### Logger Settings

Define logger (WANDB) behavior by selecting modes:

- **mode**: Choose between `online` or `offline`.
- **config**: Set your custom logging configuration via the specified configuration files.

### Model Architectures and Techniques

Switch between various GNN models and counterfactual explanation strategies. Detailed model parameters and techniques are available in:

- [config/model](config/model)
- [config/explainer](config/explainer)

## Examples

To launch the software with a specific dataset, logger mode, and run mode, use the following command:

```bash
python main.py dataset=planetoid dataset.name=cora logger.mode=online run_mode=run
```

This command will load the Cora dataset, set the logger to online mode, and run the software.

## Run the experiments
To run the experiments you can run the .sh files contained in the folder, namely :
- run_graph_experiments.sh
- run_graph_time_experiments.sh
- run_node_experiments.sh
- run_node_time_experiments.sh

The first two run the experiments on graphs classification, the other two on node classification.

The "time" experiments are run sequentially, avoiding multiprocessing to get the real explainer time.

***This framework makes extensive use of multiprocessing! Check the configuration to select how many agents and workers you can have*** 

***IMPORTANT: To get the results for different models you MUST change the variale model within the scripts with "gcn" "cheb" "graph"***

## Known Issues

- For versions of WandB greater than 0.17.5, there may be problems with multiple agents leading to errors like:

```
wandb.sdk.lib.service_connection.WandbServiceNotOwnedError: Cannot tear down service started by different process
wandb: ERROR 
```

Happy experimenting with COMBINEX!
- Sometimes AIDS and ENZYMES datasets for node classification have problems. To solve that we advice to remove the processed dataset from the processed folder. We will solve the issue asap.

