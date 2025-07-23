#!/bin/bash

# Define the output CSV file
RESULTS_CSV="gpu_results_table_custom.csv"

# Write the header row for the new table structure
echo "Problem Size,Feasible Found,Solution Time (sec.),Setup Time (sec.)" > "$RESULTS_CSV"

# Define the m x n (retailers x products) pairs from the table
pairs=(
    "3 20"
    "4 30"
    "5 40"
    "5 50"
    "6 50"
    "6 60"
    "7 60"
    "7 70"
    "8 70"
    "8 80"
    "9 80"
    "9 90"
    "10 90"
)

# Loop through each problem size pair
for pair in "${pairs[@]}"; do
    read -r m n <<< "$pair"
    echo "--- Starting tests for Problem Size: $m x $n ---"

    # Run 5 instances with different random seeds (1 through 5)
    for seed in {1..5}; do
        problem_size="${m} x ${n}"
        echo "Running instance (seed $seed)..."

        # Run the command and capture its output
        output=$(timeout 3600 ./markshare_main -m "$m" -k "$n" -s "$seed" --gpu)
        exit_code=$?

        # --- NEW PARSING LOGIC ---
        # Default values
        feasible="No"
        sol_time="N/A"
        setup_time="N/A"

        # 1. Check if a feasible solution was found
        if echo "$output" | grep -q "Found feasible solution!"; then
            feasible="Yes"
        fi

        # 2. Extract solution time
        # Looks for the line "Solution time ...", gets the last field (e.g., "0.25s"), and removes the "s"
        sol_time=$(echo "$output" | grep "Solution time" | awk '{print $NF}' | sed 's/s$//' || echo "N/A")
        
        # 3. Extract setup time
        setup_time=$(echo "$output" | grep "Setup time" | awk '{print $NF}' | sed 's/s$//' || echo "N/A")

        # Handle timeouts
        if [ $exit_code -eq 124 ]; then
            echo "Instance (seed $seed) TIMED OUT at 1 hour."
            sol_time="3600.00" # Mark time as the limit
            # Feasibility might be unknown or 'No' if it timed out before finding a solution
        elif [ $exit_code -ne 0 ]; then
            echo "Instance (seed $seed) FAILED with exit code $exit_code."
            # Mark all fields as failed
            feasible="FAIL" sol_time="FAIL" setup_time="FAIL"
        else
            echo "Instance (seed $seed) finished."
        fi

        # Append the parsed results as a new line in the CSV file
        echo "$problem_size,$feasible,$sol_time,$setup_time" >> "$RESULTS_CSV"
    done
    
    echo "--- Completed all tests for $m x $n ---"
    echo ""
done

echo "✅ All experiments are complete. Results are in '$RESULTS_CSV'."