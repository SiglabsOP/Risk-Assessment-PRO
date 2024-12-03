import random
import pandas as pd
import dask
from dask import delayed
from dask.distributed import Client, progress
from multiprocessing import cpu_count, Manager
import time
import tkinter as tk
import sys
from tkinter import messagebox, Tk, Button, Label, StringVar, Frame, ttk, scrolledtext
import json
import threading
import os
import psutil
import subprocess
import logging
from dask_tasks import generate_single_batch_dask
 
 
print("Current directory:", os.getcwd())

 
from tqdm import tqdm
from risk_calculations import (
    monte_carlo_risk_simulation,
    value_at_risk,
    conditional_value_at_risk,
    risk_parity,
    calculate_final_risk
)

# Flag to indicate if the process should be aborted
 
logging.basicConfig(level=logging.DEBUG)

 

 

  
def parallel_generate_training_data_with_dask(total_iterations, size_range, value_range, num_processes=None):
    """
    Generates training data in parallel using Dask delayed.
    """
    if num_processes is None:
        num_processes = cpu_count()  # Use all available CPU cores

    iterations_per_process = (total_iterations // num_processes) // 2  # Smaller batches
    temp_files = [f"temp_{i}.csv" for i in range(num_processes)]

    # Initialize Dask client
    client = Client()  # Start a local Dask cluster

    try:
        # Create Dask delayed tasks for parallel processing
        tasks = [
            delayed(generate_single_batch_dask)(iterations_per_process, size_range, value_range, temp_file)
            for temp_file in temp_files
        ]

        # Compute all tasks (this will start the execution)
        dask_results = client.compute(tasks)

        # Wait for all tasks to complete
        progress(dask_results)

        # Combine results from temporary files
        combined_data = pd.concat(
            [pd.read_csv(temp_file) for temp_file in temp_files if os.path.exists(temp_file)], 
            ignore_index=True
        )

    except Exception as e:
        print(f"Error during computation: {e}")
        combined_data = None  # Handle gracefully

    finally:
        # Cleanup temporary files and close client
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        client.close()

    return combined_data




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
        logging.info("Saving training data to CSV...")
        if training_data is not None:
            training_data.to_csv("training_data.csv", index=False)
            logging.info("Training data saved successfully.")
        else:
            logging.warning("No training data to save.")
    except Exception as e:
        logging.error(f"Error saving training data to CSV: {e}")

    try:
        logging.info("Saving thresholds to JSON...")
        if thresholds:
            thresholds = {key: float(value) for key, value in thresholds.items()}
            with open("risk_thresholds.json", "w") as f:
                json.dump(thresholds, f, indent=4)
            logging.info("Thresholds saved successfully.")
        else:
            logging.warning("No thresholds to save.")
    except Exception as e:
        logging.error(f"Error saving thresholds: {e}")

 

def estimate_time(iterations):
    """
    Estimate the time for the training process based on known benchmarks.
    """
    benchmark_iterations = 24331296
    benchmark_time = 50 * 60  # in seconds (50 minutes)
    estimated_time = (iterations / benchmark_iterations) * benchmark_time
    minutes, seconds = divmod(estimated_time, 60)
    return f"{int(minutes)} minutes and {int(seconds)} seconds"

def terminate_and_restart_ra_py():
    try:
        ra_running = False  # Flag to track if RA.py was found
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('RA.py' in arg for arg in cmdline):
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

class RedirectConsole:
    """
    Redirects console output to a Tkinter Text widget.
    """
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # Auto-scroll to the latest message
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass  # Required for compatibility with stdout

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Risk Training GUI")
        self.create_widgets()

    def create_widgets(self):
        # Main frame
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Output Text Area
        self.output_console = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, width=80)
        self.output_console.grid(row=0, column=0, columnspan=2, pady=(5, 10))
        self.output_console.configure(state="disabled")  # Read-only by default
        sys.stdout = RedirectConsole(self.output_console)  # Redirect stdout to GUI

        # Progress Bar
        self.progress_bar = ttk.Progressbar(frame, mode="indeterminate", length=400)
        self.progress_bar.grid(row=1, column=0, columnspan=2, pady=(5, 10))
        self.progress_bar.start()

        # Exit Button
        self.exit_button = ttk.Button(frame, text="Exit", command=self.exit_gui)
        self.exit_button.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)

    def exit_gui(self):
        """
        Gracefully exits the GUI without abrupt termination.
        """
        self.root.quit()


def start_training():
    """
    Starts the training process in a thread-safe way.
    """
    try:
        total_iterations = 24331296  # Example
        size_range = (1, 100)  # Example
        value_range = (1000, 10000)  # Example
        num_processes = 4

        print(f"Starting training with {total_iterations} iterations...")

        # Estimate time
        estimated_time = estimate_time(total_iterations)
        print(f"Estimated time for training: {estimated_time}")

        # Perform training with Dask
        training_data = parallel_generate_training_data_with_dask(
            total_iterations, size_range, value_range, num_processes
        )

        if training_data is not None:
            thresholds = calculate_thresholds(training_data)
            save_results(training_data, thresholds)

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Cleanup progress bar and console state
        print("Training process has ended.")

        
def start_thread():
    """
    Start the training process in a separate thread to prevent freezing the GUI.
    """
    threading.Thread(target=start_training, daemon=True).start()
        
def main():
    root = Tk()
    training_gui = TrainingGUI(root)

    # Start Training Button
    start_button = ttk.Button(training_gui.root, text="Start Training", command=start_thread)
    start_button.grid(row=3, column=0, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
