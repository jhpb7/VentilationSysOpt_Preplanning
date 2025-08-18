from pyomo2h5 import save_yaml, load_yaml

data = load_yaml("data/network_data/lab_raw.yml")


data.update(
    {
        "rho": {None: 1.2},
        "max_pressure": {None: 6000},
        "electric_energy_costs": {None: 0.1453 / 1000},  # in €/Wh
        "vfc_costs": {None: 1000},
        "operating_years": {None: 12},
        "operating_days_per_year": {None: 365},
        "operating_hours_per_day": {None: 24},
        "price_change_factor_electricity": {None: 1.081},
        "price_change_factor_service_maintenance": {None: 1.03},
        "interest_rate": {None: 1.07},
    }
)

# %% duct sizing

duct_sizing_data = {
    "duct_width_min": {(i, j): 0.2 for (i, j) in data["E_duct"][None]},
    "duct_width_max": {(i, j): 1.7 for (i, j) in data["E_duct"][None]},
    "duct_height_min": {(i, j): 0.2 for (i, j) in data["E_duct"][None]},
    "duct_height_max": {(i, j): 1.7 for (i, j) in data["E_duct"][None]},
}

# duct_sizing_data["duct_height_max"][("0~3", "0~4")] = 1.5
# duct_sizing_data["duct_width_max"][("0~3", "0~4")] = 1.5

duct_data = {
    # "lambda_duct": {None: 0.02},
    "duct_area_costs": {None: 50},
    "duct_resistance_coefficient": {None: 0.02},
}

duct_data.update(duct_sizing_data)

data.update(duct_data)

save_yaml("data/network_data/lab.yml", data)
