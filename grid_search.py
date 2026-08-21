import pandas as pd
from tqdm import tqdm
from model import SimulationConfig, evaluate_config


def main():
    # Define the grid of scenarios we want to test
    scenarios = []
    for n_main_A in [4, 5, 6]:
        for n_main_B in [3, 4]:
            for n_feeders_A, n_feeders_B, n_surge_A, n_surge_B in [
                (2, 1, 1, 1),
                (1, 1, 2, 1),
                (3, 2, 0, 0),
            ]:
                for underload_share_I2D, underload_share_S2I in [
                    (0.8, 0.5), (1.0, 0.5), (0.8, 1.0), (1.0, 1.0)
                ]:
                    cfg = SimulationConfig(
                        simulation_period=(365 + 60),
                        start_month='February', start_day=2,
                        interflood_start=('May', 10),
                        interflood_end=('June', 30),
                        alt_route_start=('July', 1),
                        alt_route_end=('November', 10),
                        monthly_prod_rate=[
                            3000, 3000, 3000, 3000, 3000, 3666, 3666, 3667, 3667, 3667, 3667, 3000
                        ],
                        n_containers=20_000,
                        n_feeders_A=n_feeders_A, n_feeders_B=n_feeders_B, n_feeders_C=4,
                        n_mainlands_A=n_main_A, n_mainlands_B=n_main_B,
                        n_post_interflood_A=n_surge_A, n_post_interflood_B=n_surge_B,
                        # if we disable it, then everything colapses after interflood ends...
                        disable_underloaded_I2S=False,
                        underload_share_I2D=underload_share_I2D,
                        underload_share_S2I=underload_share_S2I,
                        underload_share_S2D=0.5,
                    )
                    scenarios.append(cfg)

    n_sims = 10000
    print(f"Testing {len(scenarios)} cfgurations with {n_sims} Monte Carlo runs each...")

    results = []
    for cfg in tqdm(scenarios, desc="Evaluating Configs", unit="cfg"):
        res = evaluate_config(cfg, n_sims)
        results.append(res)

    df_results = pd.DataFrame(results)

    res_sorted = df_results.sort_values(by='avg_delivered', ascending=False)

    # Filter for viable configurations
    viable = df_results[df_results['failure_rate'] <= 0.02]

    if not viable.empty:
        viable_sorted = viable.sort_values(by='avg_delivered', ascending=False)

        # Prevent pandas from truncating columns or rows in the console
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_rows', None)

        print("\n--- VIABLE CONFIGURATIONS ---")
        # to_string() forces the full table to print without summary truncation
        print(viable_sorted.to_string(index=False)) 

        # Optional: Save to CSV if you want to open it in Excel to filter/sort manually
        # viable_sorted.to_csv("optimal_fleet_cfgs.csv", index=False)
    else:
        print("\nNo configurations achieved a 2%% failure rate")


if __name__ == '__main__':
    main()
