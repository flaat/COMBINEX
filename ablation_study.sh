#!/bin/bash

# Example list of technique names
technique_name="combinex"  # Replace with your actual technique names "cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr"  "cff" "combined"
dataset="cuneiform" 
policy="constant"
task="graph"
model="gine"
alpha=(0 1)
beta=(0 1)
lambda=(0 1)

for a in "${alpha[@]}"; do
    for b in "${beta[@]}"; do
        for l in "${lambda[@]}"; do

            if [ $a -eq 0 ] && [ $b -eq 0 ] && [ $l -eq 0 ]; then
                continue
            fi
            echo "Running with alpha=$a, beta=$b, lambda=$l"
            python main.py run_mode=sweep logger.mode=online explainer=$technique_name scheduler.policy=$policy dataset=$dataset model=$model task=graph scheduler.initial_alpha=$a explainer.beta=$b explainer.p_lambda=$l project="ABLATION-STUDY" workers=4 num_agents=4

        done
    done
done