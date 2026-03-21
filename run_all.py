
import sys
import os


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
from mpc_simulate  import run_simulation      
from plot_results  import plot_all


def main():
    print("\n" + "="*60)
    print("  Multi-Modal MPC for Highway Driving")
    print("  Vehicle Model + MPC Formulation + Cost + Constraints")
    print("="*60)

    
    out_prefix = os.path.join(_HERE, "mpc")

    results = {}

    for scenario in ['cruise', 'overtake', 'dense']:
        print(f"\n{'─'*60}")
        print(f"  Running scenario: {scenario.upper()}")
        print(f"{'─'*60}")
        data = run_simulation(
            scenario = scenario,
            T_sim    = 15.0,
            N        = 15,
            dt       = 0.1,
            verbose  = True,
        )
        plot_all(data, prefix=out_prefix)
        results[scenario] = data

    # Summary table
    print("\n" + "="*65)
    print(f"  {'SCENARIO':<10} | {'Mean v [m/s]':>12} | "
          f"{'Mean |Dy| [m]':>13} | {'Solve [ms]':>10}")
    print("-"*65)
    for sc, data in results.items():
        sv = data['states'][:, 3].mean()
        sy = abs(data['states'][:, 1] - data['y_ref']).mean()
        st = data['solve_times'].mean() * 1000
        print(f"  {sc.upper():<10} | {sv:>12.2f} | {sy:>13.3f} | {st:>8.1f} ms")
    print("="*65)
    print(f"\nAll plots saved to: {_HERE}/")


if __name__ == "__main__":
    main()