# Risk Assessment PRO
Risk Assessment PRO 101.109

# Risk Assessment and Training Software 

## Overview

This repository contains a **Risk Assessment** and **Training Software** designed to simulate and calculate trade risks using various financial models. It supports both risk analysis and training modes, where data is generated and used to calculate thresholds for different risk levels. Use the Training functionality to teach the program optimal thresholds.

The program is designed to:
- Simulate trade data using Monte Carlo simulations and other risk models.
- Calculate risk metrics such as **Monte Carlo Risk**, **Value at Risk (VaR)**, **Conditional VaR (CVaR)**, and **Risk Parity**.
- Generate training data and calculate thresholds for **Low**, **Medium**, and **High** risk categories.
- Operate efficiently with multi-core processors by utilizing parallel computation for training data generation.
- Provide a user interface built with `Tkinter` to control and monitor the process.
- the optional Training mode uses Dask and can hence handle large bulk datas. Let it run for an hour for extra accuracy, however it runs slow, it has GUI
- the standard training mode is fast , you can adjust values if needed.

## Features

- **Risk Assessment**: Calculate multiple risk metrics for simulated trades.
- **Parallel Data Generation**: Utilize multiple CPU cores for faster training data generation.
- **Training Mode**: Generate large datasets of trade simulations to calculate risk thresholds.
- **Abort Functionality**: Allow users to abort the training process at any time.
- **RA.py Restart**: Automatically restart the main Risk Assessment software after training completion.
- **User Interface**: Interactive interface using `Tkinter` for monitoring and controlling the training process.

While this program does not define risk for you directly, it can help you set reference points and help you manage risk indirectly.
 
![Screenshot 2024-12-02 014535](https://github.com/user-attachments/assets/3505b79b-6f6a-4875-affc-8913041716c4)
![Screenshot 2024-12-02 014530](https://github.com/user-attachments/assets/c01fd6c7-979b-49ba-bb4d-4ac926b62048)

If you enjoy this program, buy me a coffee https://buymeacoffee.com/siglabo
You can use it free of charge or build upon my code. 
 
(c) Peter De Ceuster 2024
Software Distribution Notice: https://peterdeceuster.uk/doc/code-terms 
This software is released under the FPA General Code License.

 
