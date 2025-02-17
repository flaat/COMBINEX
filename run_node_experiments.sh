#!/bin/bash

# Example list of technique names
technique_names=("cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr")  # Replace with your actual technique names "cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr"
dataset=( "karate" "actor" "wiki" "facebook" "cora" "citeseer" "pubmed" "AIDS" "texas" "wisconsin" "cornell" "protein"  "karate" "actor")
policies=("linear" "exponential" "sinusoidal" "dynamic" "constant")
model="graph"
for data in "${dataset[@]}"; do
    echo "Running with technique: $data"

    # Loop over each technique name
    for technique_name in "${technique_names[@]}"; do

        if [ "$technique_name" == "combined" ]; then
            # Loop over each scheduler policy for the "combined" technique
            for policy in "${policies[@]}"; do
                echo "Running with technique: $technique_name and scheduler policy: $policy"
                
                # Run the Python command with the scheduler.policy argument
                python main.py run_mode=sweep logger.mode=online explainer=$technique_name scheduler.policy=$policy dataset=$data model=$model task=node project="COMBINEX-NODE"
            done
        else
            echo "Running with technique: $technique_name"
            
            # Run the Python command without scheduler.policy
            python main.py run_mode=sweep logger.mode=online explainer=$technique_name dataset=$data model=$model task=node project="COMBINEX-NODE"
        fi
    done
done