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

logging.basicConfig(level=logging.DEBUG)

def parallel_generate_training_data_with_dask(total_iterations, size_range, value_range, num_processes=None):
    if num_processes is None:
        num_processes = cpu_count()  # Use all available CPU cores

    iterations_per_process = (total_iterations // num_processes) // 2  # Smaller batches
    temp_files = [f"temp_{i}.csv" for i in range(num_processes)]

    client = Client()  # Start a local Dask cluster

    try:
        tasks = [
            delayed(generate_single_batch_dask)(iterations_per_process, size_range, value_range, temp_file)
            for temp_file in temp_files
        ]

        dask_results = client.compute(tasks)

        progress(dask_results)

        combined_data = pd.concat(
            [pd.read_csv(temp_file) for temp_file in temp_files if os.path.exists(temp_file)], 
            ignore_index=True
        )

    except Exception as e:
        print(f"Error during computation: {e}")
        combined_data = None

    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        client.close()

    return combined_data

def calculate_thresholds(results_df):
    thresholds = {
        "Low": results_df["Final Risk"].quantile(0.3),
        "Medium": results_df["Final Risk"].quantile(0.7),
        "High": results_df["Final Risk"].max()
    }
    return thresholds

def save_results(training_data, thresholds):
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
    benchmark_iterations = 400  # Number of iterations in the benchmark
    benchmark_time = 40  # Time for 400 iterations in seconds
    
    # Calculate time per iteration
    time_per_iteration = benchmark_time / benchmark_iterations
    
    # Estimate the total time for the given number of iterations
    estimated_time = iterations * time_per_iteration
    minutes, seconds = divmod(estimated_time, 60)
    
    return f"{int(minutes)} minutes and {int(seconds)} seconds"


def terminate_and_restart_ra_py():
    try:
        ra_running = False
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('RA.py' in arg for arg in cmdline):
                ra_running = True
                proc.terminate()
                print(f"RA.py (PID: {proc.info['pid']}) terminated successfully.")
                break

        if not ra_running:
            print("RA.py not found running.")

        subprocess.Popen(["python", "RA.py"])
        print("RA.py restarted successfully.")
    except psutil.NoSuchProcess as e:
        print(f"Process already terminated: {e}")
    except psutil.AccessDenied as e:
        print(f"Permission denied when attempting to terminate RA.py: {e}")
    except Exception as e:
        print(f"Error terminating or restarting RA.py: {e}")

class RedirectConsole:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.after(0, self._write, message)

    def _write(self, message):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Risk Assessment Training (c) 2024 SIG Labs ")
        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.output_console = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, width=80)
        self.output_console.grid(row=0, column=0, columnspan=2, pady=(5, 10))
        self.output_console.configure(state="disabled")
        sys.stdout = RedirectConsole(self.output_console)

        self.progress_bar = ttk.Progressbar(frame, mode="indeterminate", length=400)
        self.progress_bar.grid(row=1, column=0, columnspan=2, pady=(5, 10))
        self.progress_bar.start()

        self.exit_button = ttk.Button(frame, text="Abort and Exit", command=self.exit_gui)
        self.exit_button.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=5)

    def update_progress_bar(self, progress_value):
        self.progress_bar["value"] = progress_value
        self.root.update_idletasks()

    def exit_gui(self):
        self.root.quit()

def start_training(training_gui):
    try:
        total_iterations = 36000
        size_range = (1, 100)
        value_range = (1000, 10000)
        num_processes = 4

        estimated_time = estimate_time(total_iterations)
        print(f"Estimated time for training: {estimated_time}")

        for i in tqdm(range(total_iterations)):
            if i % 1000 == 0:
                progress_value = (i / total_iterations) * 100
                training_gui.update_progress_bar(progress_value)

            time.sleep(0.1)

        training_data = parallel_generate_training_data_with_dask(
            total_iterations, size_range, value_range, num_processes
        )

        if training_data is not None:
            thresholds = calculate_thresholds(training_data)
            save_results(training_data, thresholds)
            print("Training completed successfully.")

    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        training_gui.update_progress_bar(100)
        print("Training process has ended.")

def start_thread(training_gui):
    training_thread = threading.Thread(target=start_training, args=(training_gui,), daemon=True)
    training_thread.start()

def main():
    root = Tk()
    training_gui = TrainingGUI(root)

    start_button = ttk.Button(training_gui.root, text="Start Training", command=lambda: start_thread(training_gui))
    start_button.grid(row=3, column=0, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
