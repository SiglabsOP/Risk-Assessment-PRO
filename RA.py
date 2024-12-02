import tkinter as tk
from tkinter import ttk, messagebox
from tkcalendar import Calendar
import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import json
import subprocess
import webbrowser



# Set up colors for a modern, enterprise-style GUI
PRIMARY_COLOR = "#1E3A8A"  # Enterprise blue
SECONDARY_COLOR = "#003366"  # Dark blue
DATA_FILE = "trade_data.json"

 

    
def monte_carlo_risk_simulation(trade_size, trade_value, num_simulations=10000):
    if trade_value <= 0 or trade_size <= 0:
        return 0

    # Adjusted ranges for simulation
    mean_return = np.random.uniform(-0.05, 0.05)  # Between -5% and +5%
    volatility = np.random.uniform(0.05, 0.25)   # Between 5% and 25%

    # Generate random simulations
    simulations = np.random.normal(
        loc=trade_value * mean_return,
        scale=trade_value * volatility,
        size=num_simulations
    )

    # Extract losses
    losses = simulations[simulations < 0]
    if len(losses) == 0:
        return 0.01

    # Calculate the 95th percentile of losses
    percentile_loss = -np.percentile(losses, 95)

    # Normalize risk factor
    normalized_risk = percentile_loss / (trade_value * volatility)
    final_risk_factor = min(max(normalized_risk, 0.01), 1)  # Clamp between 0.01 and 1

    return final_risk_factor




    
    
    
    
def value_at_risk(trade_value, confidence_level=0.95, num_simulations=10000):
    if trade_value <= 0:
        return 0
    # Simulate losses
    losses = np.random.normal(
        loc=trade_value * 0.01,
        scale=trade_value * np.random.uniform(0.03, 0.07),
        size=num_simulations
    )
    # Use a less aggressive threshold for losses
    var_threshold = np.percentile(losses, (1 - confidence_level) * 100)
    # Normalize by trade value
    var = abs(var_threshold) / trade_value
    return min(max(var, 0), 1)  # Clamp to [0, 1]



def conditional_value_at_risk(trade_value, confidence_level=0.95, num_simulations=10000):
    if trade_value <= 0:
        return 0
    losses = np.random.normal(
        loc=trade_value * 0.01,
        scale=trade_value * np.random.uniform(0.03, 0.07),
        size=num_simulations
    )
    losses_sorted = np.sort(losses)
    var_index = int((1 - confidence_level) * len(losses_sorted))
    cvar = np.mean(losses_sorted[:var_index]) if var_index > 0 else 0
    # Normalize CVaR
    normalized_cvar = abs(cvar) / trade_value
    return min(max(normalized_cvar, 0), 1)  # Clamp to [0, 1]




def risk_parity(trade_value):
    if trade_value <= 0:
        return 0
    # Adjust risk parity based on a more dynamic model or market conditions
    risk_parity_value = trade_value * np.random.uniform(0.05, 0.15) / (trade_value + 10)
    return min(max(risk_parity_value, 0), 1)  # Clamp to [0, 1]


    

# Data Handling
def save_data(trades):
    try:
        trades.to_json(DATA_FILE, orient="records", date_format="iso")
    except Exception as e:
        print(f"Error saving data: {e}")

def load_data():
    if os.path.exists(DATA_FILE):
        try:
            trades = pd.read_json(DATA_FILE)
            # Ensure all required columns are present
            required_columns = ["Date", "Ticker", "Trade Size", "Trade Value", "Monte Carlo Risk", "VaR", "CVaR", "Risk Parity", "Final Risk Factor"]
            for col in required_columns:
                if col not in trades.columns:
                    trades[col] = np.nan
            return trades
        except Exception as e:
            print(f"Error loading data: {e}")
    # Return empty DataFrame with required columns
    return pd.DataFrame(columns=["Date", "Ticker", "Trade Size", "Trade Value", "Monte Carlo Risk", "VaR", "CVaR", "Risk Parity", "Final Risk Factor"])


class TradeTrackerApp:

 
    def format_risk_level(self, risk):
        """
        Determine the risk level based on thresholds. Use default values if the file is not found.
        """
        default_thresholds = {"Low": 0.08, "Medium": 0.12, "High": float("inf")}  # Default thresholds
    
        try:
            with open("risk_thresholds.json", "r") as file:
                thresholds = json.load(file)
        except FileNotFoundError:
            thresholds = default_thresholds  # Fallback to defaults if file not found
    
        if risk < thresholds['Low']:
            return "Low", "green"
        elif risk < thresholds['Medium']:
            return "Medium", "orange"
        else:
            return "High", "red"

    def refresh_risk_levels(self):
        """
        Refresh risk levels for all entries in the history tree based on the latest thresholds.
        """
        for item in self.history_tree.get_children():
            values = self.history_tree.item(item, "values")
            final_risk = float(values[8])  # Assuming final risk is in the 9th column
            risk_label, color = self.format_risk_level(final_risk)
            self.history_tree.item(item, values=values[:-1] + (risk_label,))
            self.history_tree.tag_configure(risk_label, background=color, foreground="white")

    def add_trade_to_history(self, date, ticker, size, value, monte_carlo, var, cvar, risk_parity, final_risk):
        """
        Add a trade to the history tree with a dynamically determined risk level.
        """
        # Ensure the date is in the desired format
        if isinstance(date, pd.Timestamp):  # Check if it's a Timestamp
            formatted_date = date.strftime('%Y-%m-%d')
        elif isinstance(date, str):  # If it's already a string
            formatted_date = datetime.strptime(date, '%Y-%m-%d').strftime('%Y-%m-%d')
        else:
            formatted_date = str(date)  # Fallback to string representation
    
        risk_label, color = self.format_risk_level(final_risk)
        new_item = self.history_tree.insert(
            "",
            "end",
            values=(
                formatted_date, ticker, size, value, f"{monte_carlo:.2f}",
                f"{var:.2f}", f"{cvar:.2f}", f"{risk_parity:.2f}", f"{final_risk:.2f}", risk_label
            ),
            tags=(risk_label,)
        )
        self.history_tree.tag_configure("Low", background="green", foreground="white")
        self.history_tree.tag_configure("Medium", background="orange", foreground="white")
        self.history_tree.tag_configure("High", background="red", foreground="white")
        self.history_tree.see(new_item)  # Automatically scroll to the last entry

 
 
    
    def generate_report(self):
        # Create an HTML report
        report_path = os.path.join(os.getcwd(), "risk_report.html")
        try:
            with open(report_path, "w") as report_file:
                report_file.write("<html><head><title>Risk Assessment Report (c) SIG Labs</title></head><body>")
                report_file.write("<h1 style='text-align:center; color:#003366;'>Risk Assessment Report (c) SIG 2024</h1>")
                report_file.write("<table border='1' style='width:100%; text-align:center;'>")
                report_file.write("<tr><th>Date</th><th>Ticker</th><th>Size</th><th>Value</th>"
                                  "<th>Monte Carlo</th><th>VaR</th><th>CVaR</th><th>Risk Parity</th><th>Final Risk</th></tr>")
                
                for _, row in self.trades.iterrows():   
                        risk_level, _ = self.format_risk_level(row['Final Risk Factor'])
                        report_file.write(f"<tr><td>{row['Date']}</td><td>{row['Ticker']}</td><td>{row['Trade Size']}</td>"
                                        f"<td>{row['Trade Value']}</td><td>{row['Monte Carlo Risk']:.2f}</td>"
                                        f"<td>{row['VaR']:.2f}</td><td>{row['CVaR']:.2f}</td>"
                                        f"<td>{row['Risk Parity']:.2f}</td><td>{row['Final Risk Factor']:.2f}</td>"
                                        f"<td>{risk_level}</td></tr>")
                
                report_file.write("</table></body></html>")
            
            # Automatically open the report in a browser
            os.startfile(report_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")

            
 
    
    def create_about_tab(self, frame):
        # Create header for the About tab
        header = tk.Frame(frame, bg=SECONDARY_COLOR, height=50)
        header.pack(fill="x")
        title = tk.Label(header, text="About", bg=SECONDARY_COLOR, fg="white", font=("Helvetica", 18, "bold"))
        title.pack(pady=10)
    
        # Create about message with website and copyright info
        about_message = """
        Welcome to Risk Assessment PRO v 101.109 , the Risk Analysis Tool!
    
        This application is designed to simulate trade risks using various methods such as
        Monte Carlo simulations, Value at Risk (VaR), Conditional Value at Risk (CVaR), and Risk Parity.
        It helps traders assess potential risks and make informed decisions.
    
        For more information and updates, visit our website.
    
  
    
        (c) Peter De Ceuster
        """
    
        # Create a Text widget to display the about message
        about_text = tk.Text(frame, wrap="word", font=("Helvetica", 12), height=15)
        about_text.insert("1.0", about_message)
        about_text.config(state="disabled")
        about_text.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Create a clickable label for the website link
        website_label = tk.Label(frame, text="https://peterdeceuster.uk/", fg="blue", cursor="hand2", font=("Helvetica", 12))
        website_label.pack(pady=10)
    
        # Bind the label to open the website in the default web browser when clicked
        website_label.bind("<Button-1>", lambda e: webbrowser.open("https://peterdeceuster.uk/"))
        
    

    def __init__(self, root):
        self.root = root
        self.root.title("Risk Assessment PRO v 101.109")
        self.root.configure(bg=PRIMARY_COLOR)

        # Get screen dimensions
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()

        # Scale the window size dynamically (e.g., 80% of the screen size)
        app_width = int(screen_width * 0.8)
        app_height = int(screen_height * 0.8)

        # Center the window on the screen
        x_offset = (screen_width - app_width) // 2
        y_offset = (screen_height - app_height) // 2
        self.root.geometry(f"{app_width}x{app_height}+{x_offset}+{y_offset}")
        self.root.state("zoomed")

        # Load trade data
        self.trades = load_data()
        self.highest_trade = {"size": 0, "value": 0}

        if not self.trades.empty:
            highest_trade_row = self.trades.iloc[self.trades["Trade Value"].idxmax()]
            self.highest_trade = {
                "size": highest_trade_row["Trade Size"],
                "value": highest_trade_row["Trade Value"],
                "monte_carlo": highest_trade_row["Monte Carlo Risk"],
                "var": highest_trade_row["VaR"],
                "cvar": highest_trade_row["CVaR"],
                "risk_parity": highest_trade_row["Risk Parity"]
            }

        # Create GUI
        self.create_widgets()

        # Update visualization on startup
        self.update_visualization()
        self.refresh_risk_levels()


        # Bind window close event to save data
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        
    def adjust_slider(self, slider, delta):
        """
        Adjust the slider value by a specified delta.
        :param slider: The tk.Scale slider widget to adjust.
        :param delta: The amount to adjust the slider by (+1 or -1).
        """
        current_value = slider.get()
        new_value = current_value + delta
        slider.set(max(slider.cget("from"), min(slider.cget("to"), new_value)))

        

    def create_widgets(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)
    
        main_tab = ttk.Frame(notebook)
        vis_tab = ttk.Frame(notebook)
        help_tab = ttk.Frame(notebook)
        about_tab = ttk.Frame(notebook)  # Add the About tab frame
    
        # Add tabs to the notebook
        notebook.add(main_tab, text="Trade Tracker")
        notebook.add(vis_tab, text="Visualization")
        notebook.add(help_tab, text="Help")
        notebook.add(about_tab, text="About")  # Add the About tab
    
        # Create widgets for each tab
        self.create_main_tab_widgets(main_tab)
        self.create_visualization_tab_widgets(vis_tab)
        self.create_help_tab_widgets(help_tab)
        self.create_about_tab(about_tab)  # Create the About tab widgets

    def create_main_tab_widgets(self, frame):
        # Main horizontal split frame
        main_frame = tk.Frame(frame, bg=PRIMARY_COLOR)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Left frame for the form
        form_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        form_frame.pack(side="left", fill="y", expand=True, padx=10)
    
        # Right frame for the history tree
        history_frame = tk.Frame(main_frame, bg=PRIMARY_COLOR)
        history_frame.pack(side="right", fill="both", expand=True, padx=10)
    
        # Form elements in the left frame (same as before)
        ttk.Label(form_frame, text="Ticker Symbol:", background=PRIMARY_COLOR, foreground="white").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.ticker_entry = ttk.Entry(form_frame, font=('Helvetica', 14))
        self.ticker_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
    
        ttk.Label(form_frame, text="Trade Date:", background=PRIMARY_COLOR, foreground="white").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.calendar = Calendar(form_frame, selectmode='day', date_pattern='yyyy-mm-dd', font=('Helvetica', 12))
        self.calendar.grid(row=1, column=1, padx=10, pady=5)
    
        ttk.Label(form_frame, text="Trade Size:", background=PRIMARY_COLOR, foreground="white").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.size_slider = tk.Scale(
            form_frame,
            from_=1,
            to=1000,
            orient="horizontal",
            resolution=1,
            command=self.update_size_value,
            bg=PRIMARY_COLOR,
            fg="white",
            highlightthickness=0,
            troughcolor=SECONDARY_COLOR
        )
        self.size_slider.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        self.size_value_label = tk.Label(form_frame, text="1", background=PRIMARY_COLOR, foreground="white")
        self.size_value_label.grid(row=2, column=2, padx=10, pady=5)
        
        
        ttk.Button(form_frame, text="-", command=lambda: self.adjust_slider(self.size_slider, -1)).grid(row=2, column=3, padx=5, pady=5)    
        ttk.Button(form_frame, text="+", command=lambda: self.adjust_slider(self.size_slider, 1)).grid(row=2, column=4, padx=5, pady=5)        
        
        
    
        ttk.Label(form_frame, text="Trade Value:", background=PRIMARY_COLOR, foreground="white").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.value_slider = tk.Scale(
            form_frame,
            from_=1,
            to=50000,
            orient="horizontal",
            resolution=1,
            command=self.update_value,
            bg=PRIMARY_COLOR,
            fg="white",
            highlightthickness=0,
            troughcolor=SECONDARY_COLOR
        )
        self.value_slider.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        self.value_value_label = tk.Label(form_frame, text="1", background=PRIMARY_COLOR, foreground="white")
        self.value_value_label.grid(row=3, column=2, padx=10, pady=5)
        
        
        
        
        ttk.Button(form_frame, text="-", command=lambda: self.adjust_slider(self.value_slider, -1)).grid(row=3, column=3, padx=5, pady=5)   
        ttk.Button(form_frame, text="+", command=lambda: self.adjust_slider(self.value_slider, 1)).grid(row=3, column=4, padx=5, pady=5)        
        
    
        submit_button = ttk.Button(form_frame, text="Submit Trade", command=self.submit_trade)
        submit_button.grid(row=4, column=0, columnspan=2, pady=10)
    
        report_button = ttk.Button(form_frame, text="Generate Report", command=self.generate_report)
        report_button.grid(row=5, column=0, columnspan=2, pady=10)
        
            # Add "Training Mode" button
        training_button = ttk.Button(form_frame, text="Training Mode", command=self.launch_training_mode)
        training_button.grid(row=6, column=0, columnspan=2, pady=10)
    
        # History tree in the right frame
        self.tree_scrollbar = ttk.Scrollbar(history_frame)
        self.tree_scrollbar.pack(side="right", fill="y")
    
        self.tree_hscrollbar = ttk.Scrollbar(history_frame, orient="horizontal")
        self.tree_hscrollbar.pack(side="bottom", fill="x")
    
        self.history_tree = ttk.Treeview(
            history_frame,
            columns=("Date", "Ticker", "Size", "Value", "Monte Carlo", "VaR", "CVaR", "Risk Parity", "Final Risk", "Risk Level"),
            show="headings",
            yscrollcommand=self.tree_scrollbar.set,
            xscrollcommand=self.tree_hscrollbar.set
        )
        self.tree_scrollbar.config(command=self.history_tree.yview)
        self.tree_hscrollbar.config(command=self.history_tree.xview)
    
        # Update column widths
        column_widths = {
            "Date": 150,
            "Ticker": 100,
            "Size": 100,
            "Value": 150,
            "Monte Carlo": 150,
            "VaR": 150,
            "CVaR": 150,
            "Risk Parity": 150,
            "Final Risk": 150,
            "Risk Level": 100
        }
        for col, width in column_widths.items():
            self.history_tree.heading(col, text=col)
            self.history_tree.column(col, width=width, anchor="center")
    
        self.history_tree.pack(fill="both", expand=True, padx=10, pady=10)
        self.load_history()
    
    def launch_training_mode(self):
        try:
            subprocess.Popen(["python", "training_mode.py"], cwd=os.getcwd())  # Launch training_mode.py in the same directory
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch training mode: {e}")

            

    def create_visualization_tab_widgets(self, frame):
        header = tk.Frame(frame, bg=SECONDARY_COLOR, height=50)
        header.pack(fill="x")
        title = tk.Label(header, text="Visualization", bg=SECONDARY_COLOR, fg="white", font=("Helvetica", 18, "bold"))
        title.pack(pady=10)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Risk Assessment Comparison")
        self.ax.set_xlabel("Risk Models")
        self.ax.set_ylabel("Risk Factor")
        self.canvas = FigureCanvasTkAgg(self.fig, frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

 
    
    def create_help_tab_widgets(self, frame):
        """
        Create the Help tab and display thresholds dynamically. Use defaults if the file is missing.
        """
        default_thresholds = {"Low": 0.08, "Medium": 0.12, "High": float("inf")}  # Default thresholds
    
        try:
            with open("risk_thresholds.json", "r") as file:
                risk_thresholds = json.load(file)
        except FileNotFoundError:
            risk_thresholds = default_thresholds  # Fallback to defaults if file not found
    
        low_threshold = risk_thresholds['Low']
        medium_threshold = risk_thresholds['Medium']
        high_threshold = risk_thresholds['High']
    
        header = tk.Frame(frame, bg=SECONDARY_COLOR, height=50)
        header.pack(fill="x")
        title = tk.Label(header, text="Help & Documentation", bg=SECONDARY_COLOR, fg="white", font=("Helvetica", 18, "bold"))
        title.pack(pady=10)
    
        # Create the help message with dynamic threshold values
        help_message = f"""
            **Risk Models and Final Risk Factor**
            1. Monte Carlo Simulation:
               - Uses random variables to simulate potential trade outcomes and assess risk.
               - Focuses on the 95th percentile of simulated losses normalized by trade volatility.
            
            2. Value at Risk (VaR):
               - Measures the potential loss at a given confidence level (default: 95%).
               - Represents the minimum loss expected in extreme cases.
            
            3. Conditional VaR (CVaR):
               - Calculates the expected average loss beyond the VaR threshold.
               - Provides a deeper understanding of tail-end risks.
            
            4. Risk Parity:
               - Allocates risk proportionally based on trade size and value.
               - Encourages balanced risk exposure across trades.
            
            **Final Risk Factor Calculation:**
            - Combines all four risk models using weighted averaging:
                Final Risk = 30% Monte Carlo + 30% VaR + 20% CVaR + 20% Risk Parity.
            
            **Latest Risk Thresholds generated from training:**
            - Low Risk: Below {low_threshold:.2f}
            - Medium Risk: Between {low_threshold:.2f} and {medium_threshold:.2f}
            - High Risk: Above {medium_threshold:.2f}
            ADVICE:   Consider adjusting trade size or value if the final risk exceeds {low_threshold:.2f}.
            
            **Tips:**
            - Train on regular basis, so the Thresholds will update. The Threshold values mentioned above, will auto-update after training.
            - It is important to understand the algo values, and the low,medium or high signals, do not calculate risk. They do not directly relate to risk.              
              They are mere computated reference points to help you, calculate the risk you are willing to take trading markets.
            - Use a combination of large and small trades to observe risk variability.
            - add data or train the program before using visualization features.
            
        """
        
        # Create a frame to hold the scrollbars and text widget
        text_frame = tk.Frame(frame)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
        # Create the vertical scrollbar
        scrollbar_y = tk.Scrollbar(text_frame, orient="vertical")
    
        # Create the horizontal scrollbar
        scrollbar_x = tk.Scrollbar(text_frame, orient="horizontal")
    
        # Create the Text widget
        help_text = tk.Text(text_frame, wrap="word", font=("Helvetica", 12), yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        help_text.insert("1.0", help_message)
        help_text.config(state="disabled")
        help_text.pack(side="left", fill="both", expand=True)
    
        # Attach the scrollbars to the Text widget
        scrollbar_y.config(command=help_text.yview)
        scrollbar_y.pack(side="right", fill="y")
        
        scrollbar_x.config(command=help_text.xview)
        scrollbar_x.pack(side="bottom", fill="x")

    def update_size_value(self, val):
        self.size_value_label.config(text=f"{int(float(val))}")
        self.update_calculations() 
    
    def update_value(self, val):
        self.value_value_label.config(text=f"{int(float(val))}")
        self.update_calculations()
        
    def update_calculations(self):
            trade_size = float(self.size_slider.get())
            trade_value = float(self.value_slider.get())
            
            monte_carlo_risk = monte_carlo_risk_simulation(trade_size, trade_value)
            var = value_at_risk(trade_value)
            cvar = conditional_value_at_risk(trade_value)
            risk_parity_value = risk_parity(trade_value)
    
            # Corrected line: Call the method using self
            final_risk = self.calculate_final_risk(monte_carlo_risk, var, cvar, risk_parity_value)
    
            # Update UI or internal state as needed
            # You may want to display final_risk or use it elsewhere
    
    def calculate_final_risk(self, monte_carlo_risk, var, cvar, risk_parity_value):
        # Assign weights to each risk model
        weights = {
            "monte_carlo": 0.3,
            "var": 0.3,
            "cvar": 0.2,
            "risk_parity": 0.2
        }
        combined_risk = (
            monte_carlo_risk * weights["monte_carlo"] +
            var * weights["var"] +
            cvar * weights["cvar"] +
            risk_parity_value * weights["risk_parity"]
        )
        return min(max(combined_risk, 0), 1)  # Clamp to [0, 1]

    def submit_trade(self):
        ticker = self.ticker_entry.get()
        trade_date = self.calendar.get_date()
        trade_size = float(self.size_slider.get())

        trade_value = float(self.value_slider.get())

        if not ticker:
            messagebox.showerror("Error", "Please enter a ticker symbol.")
            return

        # Fixes for better risk calculation
        monte_carlo_risk = monte_carlo_risk_simulation(trade_size, trade_value)
        var = value_at_risk(trade_value)
        cvar = conditional_value_at_risk(trade_value)
        risk_parity_value = risk_parity(trade_value)

        # Final risk score
        risk_score = max(monte_carlo_risk, var, cvar, risk_parity_value)

        # Update highest trade
        if trade_value > self.highest_trade["value"]:
            self.highest_trade = {"size": trade_size, "value": trade_value,
                                  "monte_carlo": monte_carlo_risk, "var": var,
                                  "cvar": cvar, "risk_parity": risk_parity_value}

        # Record trade
        new_trade = pd.DataFrame([[trade_date, ticker, trade_size, trade_value, monte_carlo_risk, var, cvar, risk_parity_value, risk_score]],
                                 columns=["Date", "Ticker", "Trade Size", "Trade Value", "Monte Carlo Risk", "VaR", "CVaR", "Risk Parity", "Final Risk Factor"])

        if not new_trade.empty and not new_trade.isna().all(axis=None):
            self.trades = pd.concat([self.trades, new_trade], ignore_index=True)
            self.add_trade_to_history(trade_date, ticker, trade_size, trade_value, monte_carlo_risk, var, cvar, risk_parity_value, risk_score)
            self.update_visualization()

    def load_history(self):
        for _, trade in self.trades.iterrows():
            self.add_trade_to_history(trade["Date"], trade["Ticker"], trade["Trade Size"], trade["Trade Value"],
                                      trade["Monte Carlo Risk"], trade["VaR"], trade["CVaR"], trade["Risk Parity"], trade["Final Risk Factor"])

 
        
 

        

    def update_visualization(self):
        self.fig.clear()
    
        if self.trades.empty or not all(col in self.trades.columns for col in ["Monte Carlo Risk", "VaR", "CVaR", "Risk Parity"]):
            # Show a message if there's no data to visualize
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data available for visualization", 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.set_title("Risk Assessment Visualization")
            self.canvas.draw()
            return
    
        # Subplot 1: Portfolio Risk Summary (Bar Chart)
        ax1 = self.fig.add_subplot(121)
        aggregated_risks = self.trades[["Monte Carlo Risk", "VaR", "CVaR", "Risk Parity"]].mean()
        ax1.bar(aggregated_risks.index, aggregated_risks.values, color=["#1E90FF", "#FF6347", "#3CB371", "#FFD700"])
        ax1.set_title("Average Risk by Model")
        ax1.set_ylabel("Risk Factor (Normalized)")
        ax1.set_xlabel("Risk Models")
    
        # Subplot 2: Risk Distribution (Line Chart)
        ax2 = self.fig.add_subplot(122)
        for model in ["Monte Carlo Risk", "VaR", "CVaR", "Risk Parity"]:
            self.trades[model].plot(kind="line", ax=ax2, label=model, alpha=0.7)
        ax2.set_title("Risk Distribution Over Time")
        ax2.set_xlabel("Trade Index")
        ax2.set_ylabel("Risk Factor (Normalized)")
        ax2.legend(loc="upper left")
    
        self.fig.tight_layout()
        self.canvas.draw()



    def on_close(self):
        save_data(self.trades)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TradeTrackerApp(root)
    root.mainloop()     