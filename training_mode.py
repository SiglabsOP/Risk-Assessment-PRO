import random
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
from tkinter import messagebox, Tk, Button
import json
import threading
import os
import psutil
import subprocess
from tqdm import tqdm
from risk_calculations import (
    monte_carlo_risk_simulation,
    value_at_risk,
    conditional_value_at_risk,
    risk_parity,
    calculate_final_risk
)

# Flag to indicate if the process should be aborted
abort_training = False

def generate_single_batch(num_iterations, size_range, value_range):
    """
    Generates a batch of training data by simulating trades and calculating risks.
    """
    results = []
    for _ in tqdm(range(num_iterations), desc="Generating batches", ncols=100):
        if abort_training:
            break  # Exit if abort flag is set
        trade_size = random.randint(*size_range)
        trade_value = random.randint(*value_range)

        monte_carlo = monte_carlo_risk_simulation(trade_size, trade_value)
        var = value_at_risk(trade_value)
        cvar = conditional_value_at_risk(trade_value)
        risk_parity_value = risk_parity(trade_value)
        final_risk = calculate_final_risk(monte_carlo, var, cvar, risk_parity_value)

        results.append({
            "Trade Size": trade_size,
            "Trade Value": trade_value,
            "Monte Carlo": monte_carlo,
            "VaR": var,
            "CVaR": cvar,
            "Risk Parity": risk_parity_value,
            "Final Risk": final_risk
        })
    return results


def parallel_generate_training_data(total_iterations, size_range, value_range, num_processes=None):
    """
    Generates training data in parallel using multiprocessing.
    """
    if num_processes is None:
        num_processes = cpu_count()  # Use all available CPU cores

    iterations_per_process = total_iterations // num_processes

    with Pool(processes=num_processes) as pool:
        # Create tasks for each process
        tasks = [
            pool.apply_async(generate_single_batch, args=(iterations_per_process, size_range, value_range))
            for _ in range(num_processes)
        ]

        # Collect results from all processes and update progress bar
        all_results = []
        with tqdm(total=total_iterations, desc="Overall Progress", ncols=100) as pbar:
            for task in tasks:
                if abort_training:
                    break  # Exit if abort flag is set
                results = task.get()
                all_results.extend(results)
                pbar.update(len(results))  # Update progress bar with the number of results processed

    # Convert results into a DataFrame
    return pd.DataFrame(all_results)


def calculate_thresholds(results_df):
    """
    Calculates thresholds for risk levels based on the training data.
    """
    thresholds = {
        "Low": results_df["Final Risk"].quantile(0.3),
        "Medium": results_df["Final Risk"].quantile(0.7),
        "High": results_df["Final Risk"].max()
    }
    return thresholds


def save_results(training_data, thresholds):
    """
    Save the generated training data and risk thresholds to files.
    """
    try:
        print("Saving training data to CSV...")
        training_data.to_csv("training_data.csv", index=False)
        print("Training data saved successfully.")
    except Exception as e:
        print(f"Error saving training data to CSV: {e}")

    try:
        print("Saving thresholds to JSON...")

        # Convert np.float64 values to regular Python float
        thresholds = {key: float(value) for key, value in thresholds.items()}

        # Save to JSON file
        with open("risk_thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=4)  # Use json.dump to write the dictionary in a pretty format

        print("Thresholds saved successfully.")
    except Exception as e:
        print(f"Error saving thresholds: {e}")


def abort_training_process():
    """
    Set the flag to abort the training process.
    """
    global abort_training
    abort_training = True
    messagebox.showinfo("Training Aborted", "The training process has been aborted.")


 
def terminate_and_restart_ra_py():
    try:
        ra_running = False  # Flag to track if RA.py was found
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = proc.info.get('cmdline', [])  # Use a default empty list if cmdline is None
            if cmdline and any('RA.py' in arg for arg in cmdline):  # Check if cmdline is not None before iterating
                ra_running = True
                proc.terminate()
                print(f"RA.py (PID: {proc.info['pid']}) terminated successfully.")
                break

        if not ra_running:
            print("RA.py not found running.")

        # Restart RA.py process
        subprocess.Popen(["python", "RA.py"])
        print("RA.py restarted successfully.")
    except psutil.NoSuchProcess as e:
        print(f"Process already terminated: {e}")
    except psutil.AccessDenied as e:
        print(f"Permission denied when attempting to terminate RA.py: {e}")
    except Exception as e:
        print(f"Error terminating or restarting RA.py: {e}")



def main():
    global abort_training
    # Parameters
    NUM_ITERATIONS = 500000  # Adjust based on hardware capabilities
    SIZE_RANGE = (1, 1000)
    VALUE_RANGE = (1, 50000)

    # Create a Tkinter window for user interaction
    root = Tk()
    root.withdraw()  # Hide the root window

    # Create an "Abort" button
    abort_button = Button(root, text="Abort Training", command=abort_training_process)
    abort_button.pack()

    # Warn the user about training duration
    messagebox.showinfo(
        "Training Mode",
        "Training will take approximately 1 minute. This is very CPU intensive. Please wait..."
    )

    # Start training in a separate thread to allow UI responsiveness
    def training_thread():
        global abort_training  # Use global instead of nonlocal
        start_time = time.time()
        print("Generating training data in parallel...")

        # Generate data in parallel
        training_data = parallel_generate_training_data(NUM_ITERATIONS, SIZE_RANGE, VALUE_RANGE)

        if abort_training:
            return  # Stop if the process is aborted

        # Calculate thresholds
        print("Analyzing training data...")
        thresholds = calculate_thresholds(training_data)

        # Save results
        save_results(training_data, thresholds)

        elapsed_time = time.time() - start_time
        messagebox.showinfo(
            "Training Complete",
            f"Training completed in {elapsed_time // 60:.0f} minutes and {elapsed_time % 60:.0f} seconds.\n"
            f"Thresholds:\n"
            f"Low Risk: Below {thresholds['Low']:.2f}\n"
            f"Medium Risk: {thresholds['Low']:.2f} - {thresholds['Medium']:.2f}\n"
            f"High Risk: Above {thresholds['Medium']:.2f}\n\n"
            f"Training complete, rebooting RA Software... Please close older instances if needed."
        )

        # Terminate and restart RA.py, then shut down the console
        terminate_and_restart_ra_py()
        print("Shutting down the console...")
        os._exit(0)  # Terminate Python program immediately

    # Start the training thread
    threading.Thread(target=training_thread).start()

    # Run the Tkinter event loop to display the UI and respond to the abort button
    root.mainloop()

if __name__ == "__main__":
    main()
