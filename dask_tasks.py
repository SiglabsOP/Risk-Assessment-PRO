from tqdm import tqdm
import pandas as pd
import random
from risk_calculations import (
    monte_carlo_risk_simulation,
    value_at_risk,
    conditional_value_at_risk,
    risk_parity,
    calculate_final_risk,
)

def generate_single_batch_dask(num_iterations, size_range, value_range, output_file):
    """
    Generates a batch of training data, calculates risks, and saves intermediate results to disk.
    """
    results = []
    for _ in tqdm(range(num_iterations), desc="Generating batches", ncols=100):
        trade_size = random.randint(*size_range)
        trade_value = random.randint(*value_range)

        monte_carlo = monte_carlo_risk_simulation(trade_size, trade_value)
        var = value_at_risk(trade_value)
        cvar = conditional_value_at_risk(trade_value)
        risk_parity_value = risk_parity(trade_value)
        final_risk = calculate_final_risk(monte_carlo, var, cvar, risk_parity_value)

        results.append({"Final Risk": final_risk})

    # Save intermediate results to disk
    try:
        pd.DataFrame(results).to_csv(output_file, index=False)
    except Exception as e:
        logging.error(f"Error saving batch data to {output_file}: {e}")
