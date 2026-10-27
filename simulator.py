import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# --- Configuration ---
# Seed for reproducible "random" market behavior across app reloads
np.random.seed(42)

def generate_dates(start_date, months):
    """Generates a list of monthly timestamps."""
    return [start_date + timedelta(days=30 * i) for i in range(months)]

def random_walk(steps, start_val, volatility, drift=0.0):
    """
    Generates a Geometric Brownian Motion series.
    Safeguarded with np.maximum to prevent negative economic factors.
    """
    # Generate percent changes (normal distribution)
    pct_change = np.random.normal(drift, volatility, steps)
    
    # Cumulative product to create the 'walk' path
    path = start_val * np.cumprod(1 + pct_change)
    
    # Economics Safety: Factors cannot drop below 0.1 (10% of baseline)
    return np.maximum(path, 0.1)

def seasonality_factor(months, peak_month, amplitude):
    """
    Creates a cyclical multiplier centered at 1.0.
    peak_month: 0-11 (Jan-Dec)
    """
    x = np.arange(months)
    # Cosine wave formula adjusted so the peak aligns with peak_month
    cycle = np.cos((x - peak_month) * 2 * np.pi / 12)
    return 1 + amplitude * cycle

def inject_shocks(data, probability, magnitude_range):
    """
    Injects sudden market events (spikes or crashes).
    magnitude_range: tuple (min_impact, max_impact) e.g., (-0.2, 0.2)
    """
    # 1. Decide where shocks happen
    shock_mask = np.random.rand(len(data)) < probability
    
    # 2. Determine magnitude for each shock
    low, high = magnitude_range
    magnitudes = np.random.uniform(low, high, size=len(data))
    
    # 3. Apply only where mask is True
    multipliers = np.where(shock_mask, 1 + magnitudes, 1.0)
    return data * multipliers

def calculate_equilibrium(base_cost, demand, supply, seasonality):
    """
    The Core Economic Formula:
    Price = Cost * (Demand / Supply) * Seasonality
    """
    return base_cost * (demand / supply) * seasonality

def generate_good_history(name, params, months=60):
    """
    Simulates the specific economics of a single good over time.
    """
    # Start simulation 5 years ago from today to provide history
    start_date = datetime.now() - timedelta(days=months*30)
    dates = generate_dates(start_date, months)
    
    # 1. Generate Fundamental Factors (Random Walks)
    # Costs move with inflation/materials
    cost_base = random_walk(months, params['base_cost'], params['cost_vol'], drift=0.002) 
    # Demand drifts slightly (popularity)
    demand = random_walk(months, 1.0, params['demand_vol'], drift=params['demand_drift'])
    # Supply is reactive or steady
    supply = random_walk(months, 1.0, params['supply_vol'])
    
    # 2. Apply Seasonality (Predictable cycles)
    seasonal_mult = seasonality_factor(months, params['peak_month'], params['seasonal_amp'])
    
    # 3. Inject Real-World Shocks (Unpredictable events)
    # Supply shocks (droughts, shortages) usually spike costs
    cost_base = inject_shocks(cost_base, params['shock_prob'], (0.1, 0.3)) 
    # Demand shocks (viral trends, recessions)
    demand = inject_shocks(demand, params['shock_prob'], (-0.2, 0.2))
    
    # 4. Calculate True Equilibrium Price (The "Fair" Price)
    true_price = calculate_equilibrium(cost_base, demand, supply, seasonal_mult)
    
    # 5. Generate Market Price with "Stickiness"
    # Market prices rarely jump instantly to equilibrium; they lag.
    market_price = np.zeros(months)
    market_price[0] = true_price[0]
    
    for t in range(1, months):
        alpha = params['stickiness'] # 0.0 = instant adjustment, 1.0 = never changes
        
        # The market tries to find the true price, but has noise (trader emotion)
        noise = np.random.normal(0, 0.02) # 2% noise
        target = true_price[t] * (1 + noise)
        
        # Update rule: Old Price * Alpha + New Target * (1 - Alpha)
        market_price[t] = (alpha * market_price[t-1]) + ((1 - alpha) * target)

    # 6. Formatting
    return pd.DataFrame({
        'date': dates,
        'good': name,
        'market_price': np.round(market_price, 2),
        'true_equilibrium': np.round(true_price, 2),
        'cost_factor': np.round(cost_base, 2),
        'demand_factor': np.round(demand, 2),
        'supply_factor': np.round(supply, 2),
        'seasonality': np.round(seasonal_mult, 3)
    })

def get_simulation_data():
    """
    Main entry point. Returns a combined DataFrame for all product types.
    """
    # Distinct economic profiles for realistic variety
    configs = {
        'Organic Wheat (Crop)': {
            'base_cost': 220, 
            'cost_vol': 0.03,    # High cost volatility (fertilizer/fuel)
            'demand_vol': 0.05, 
            'demand_drift': 0.001,
            'supply_vol': 0.06,  # Volatile supply (weather)
            'peak_month': 6,     # Summer harvest
            'seasonal_amp': 0.25,# Strong seasonality
            'shock_prob': 0.05,  # Frequent shocks
            'stickiness': 0.2    # Low stickiness (commodity market moves fast)
        },
        'Handmade Table (Product)': {
            'base_cost': 450, 
            'cost_vol': 0.01,    # Stable material costs
            'demand_vol': 0.03, 
            'demand_drift': 0.002,
            'supply_vol': 0.01, 
            'peak_month': 11,    # Christmas rush
            'seasonal_amp': 0.15,
            'shock_prob': 0.02,  # Rare shocks
            'stickiness': 0.5    # Medium stickiness (retail pricing)
        },
        'Studio Apt (Rent)': {
            'base_cost': 1200, 
            'cost_vol': 0.002,   # Very stable costs (mortgage/tax)
            'demand_vol': 0.02, 
            'demand_drift': 0.004, # Rising rent trend
            'supply_vol': 0.005, 
            'peak_month': 8,     # August move-ins
            'seasonal_amp': 0.05,# Low seasonality
            'shock_prob': 0.01,  # Very rare shocks
            'stickiness': 0.9    # High stickiness (12-month leases)
        }
    }

    all_data = []
    for name, params in configs.items():
        df = generate_good_history(name, params)
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

if __name__ == "__main__":
    # Self-test if run directly
    df = get_simulation_data()
    print(f"Simulation Complete. Generated {len(df)} rows.")
    print(df.groupby('good').tail(2)[['date', 'good', 'market_price', 'true_equilibrium']])
