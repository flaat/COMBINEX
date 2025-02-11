#!/bin/bash

# Example list of technique names
technique_names=("cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined")  # Replace with your actual technique names "cf-gnnfeatures" "cf-gnn"  "random-feat" "random" "ego" "cff" "combined" "unr"

# Define the scheduler policies for the "combined" technique
scheduler_policies=("linear" "exponential" "sinusoidal" "dynamic" "constant")

dataset=("aids_g" "enzymes_g" "protein_g" "coil")

for data in "${dataset[@]}"; do
    echo "Running with technique: $data"

    # Loop over each technique name
    for technique_name in "${technique_names[@]}"; do

        if [ "$technique_name" == "combined" ]; then
            # Loop over each scheduler policy for the "combined" technique
            for policy in "${scheduler_policies[@]}"; do
                echo "Running with technique: $technique_name and scheduler policy: $policy"
                
                # Run the Python command with the scheduler.policy argument
                python main.py run_mode=sweep logger.mode=online explainer=$technique_name scheduler.policy=$policy dataset=$data model=cheb
            done
        else
            echo "Running with technique: $technique_name"
            
            # Run the Python command without scheduler.policy
            python main.py run_mode=sweep logger.mode=online explainer=$technique_name dataset=$data model=cheb
        fi
    done
done