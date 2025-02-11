#!/bin/bash

# Example list of technique names
technique_names=("cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr")  # Replace with your actual technique names "cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr"
#"karate" "actor" "wiki" "facebook" "cora" "citeseer" "pubmed" "AIDS" "texas" "wisconsin" "cornell"
dataset=("enzymes" )
policies=("linear")
model="gcn"
epochs=(1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180 190 200)
for epoch in "${epochs[@]}"; do
    for data in "${dataset[@]}"; do
        echo "Running with technique: $data"

        # Loop over each technique name
        for technique_name in "${technique_names[@]}"; do

            if [ "$technique_name" == "combined" ]; then
                # Loop over each scheduler policy for the "combined" technique
                for policy in "${policies[@]}"; do
                    echo "Running with technique: $technique_name and scheduler policy: $policy"
                    
                    # Run the Python command with the scheduler.policy argument
                    python main.py run_mode=sweep logger.mode=online explainer=$technique_name scheduler.policy=$policy dataset=$data model=$model trainer.epochs=$epoch workers=3 num_agents=3 max_samples=100 project="COMBINEX-EPOCHS" logger.config=time_sweep task=node
                done
            else
                echo "Running with technique: $technique_name"
                
                # Run the Python command without scheduler.policy
                python main.py run_mode=sweep logger.mode=online explainer=$technique_name dataset=$data model=$model trainer.epochs=$epoch workers=3 num_agents=3 max_samples=100 project="COMBINEX-EPOCHS" logger.config=time_sweep task=node
            fi
        done
    done
done