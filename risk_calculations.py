import numpy as np

def monte_carlo_risk_simulation(trade_size, trade_value, num_simulations=10000):
    if trade_value <= 0 or trade_size <= 0:
        return 0

    mean_return = np.random.uniform(-0.05, 0.05)  # Between -5% and +5%
    volatility = np.random.uniform(0.05, 0.25)   # Between 5% and 25%

    simulations = np.random.normal(
        loc=trade_value * mean_return,
        scale=trade_value * volatility,
        size=num_simulations
    )

    losses = simulations[simulations < 0]
    if len(losses) == 0:
        return 0.01

    percentile_loss = -np.percentile(losses, 95)
    normalized_risk = percentile_loss / (trade_value * volatility)
    return min(max(normalized_risk, 0.01), 1)  # Clamp between 0.01 and 1

def value_at_risk(trade_value, confidence_level=0.95, num_simulations=10000):
    if trade_value <= 0:
        return 0
    losses = np.random.normal(
        loc=trade_value * 0.01,
        scale=trade_value * np.random.uniform(0.03, 0.07),
        size=num_simulations
    )
    var_threshold = np.percentile(losses, (1 - confidence_level) * 100)
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
    normalized_cvar = abs(cvar) / trade_value
    return min(max(normalized_cvar, 0), 1)  # Clamp to [0, 1]

def risk_parity(trade_value):
    if trade_value <= 0:
        return 0
    risk_parity_value = trade_value * np.random.uniform(0.05, 0.15) / (trade_value + 10)
    return min(max(risk_parity_value, 0), 1)  # Clamp to [0, 1]

def calculate_final_risk(monte_carlo, var, cvar, risk_parity):
    weights = {
        "monte_carlo": 0.3,
        "var": 0.3,
        "cvar": 0.2,
        "risk_parity": 0.2
    }
    combined_risk = (
        monte_carlo * weights["monte_carlo"] +
        var * weights["var"] +
        cvar * weights["cvar"] +
        risk_parity * weights["risk_parity"]
    )
    return min(max(combined_risk, 0), 1)  # Clamp to [0, 1]
