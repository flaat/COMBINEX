#!/bin/bash

# Example list of technique names
technique_names=( "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr")  # Replace with your actual technique names "cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr"
dataset=( "protein"  "enzymes"  "karate" "actor" "wiki" "facebook" "cora" "citeseer" "pubmed" "AIDS" "texas" "wisconsin" "cornell" )
policies=("linear" )

model="gcn"
for data in "${dataset[@]}"; do
    echo "Running with technique: $data"

    # Loop over each technique name
    for technique_name in "${technique_names[@]}"; do

        if [ "$technique_name" == "combined" ]; then
            # Loop over each scheduler policy for the "combined" technique
            for policy in "${policies[@]}"; do
                echo "Running with technique: $technique_name and scheduler policy: $policy"
                
                # Run the Python command with the scheduler.policy argument
                python main.py run_mode=sweep logger.mode=online explainer=$technique_name scheduler.policy=$policy dataset=$data model=$model workers=1 num_agents=1 max_samples=80 project="COMBINEX-TIME-NODE" logger.config=time_sweep task=node
            done
        else
            echo "Running with technique: $technique_name"
            
            # Run the Python command without scheduler.policy
            python main.py run_mode=sweep logger.mode=online explainer=$technique_name dataset=$data model=$model  workers=1 num_agents=1 max_samples=80 project="COMBINEX-TIME-NODE" logger.config=time_sweep task=node
        fi
    done
done