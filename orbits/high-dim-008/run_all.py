#!/usr/bin/env python3
"""Run all high-D experiments and save results to JSON.

Usage:
    cd /Users/wujiewang/code/det-sampler/.worktrees/high-dim-008
    PYTHONPATH=. python orbits/high-dim-008/run_all.py [--quick] [--budget N]
"""
import sys, os, json, time
import importlib.util

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_this_dir, '..', '..'))
sys.path.insert(0, _project_root)

# Import solution.py via importlib (can't use normal import due to hyphens)
spec = importlib.util.spec_from_file_location(
    "solution", os.path.join(_this_dir, "solution.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        budget = 500_000
    elif args.budget:
        budget = args.budget
    else:
        budget = None

    if budget:
        for exp in S.EXPERIMENTS.values():
            exp['n_force_evals'] = min(exp['n_force_evals'], budget)

    results_json = {}
    results_plots = {}

    for exp_name in S.EXPERIMENTS:
        results_json[exp_name] = {}
        results_plots[exp_name] = {}
        print(f'\n{"="*60}', flush=True)
        print(f'System: {exp_name}', flush=True)
        print(f'{"="*60}', flush=True)

        for sname in S.SAMPLER_NAMES:
            try:
                r = S.run_experiment(exp_name, sname, seed=args.seed)
                results_plots[exp_name][sname] = r
                metrics = {k: v for k, v in r.items()
                          if k not in ('energies', 'q_traj', 'p_traj')}
                for k, v in list(metrics.items()):
                    if hasattr(v, 'tolist'):
                        metrics[k] = v.tolist()
                results_json[exp_name][sname] = metrics
            except Exception as e:
                print(f'  ERROR {sname}: {e}', flush=True)
                import traceback; traceback.print_exc()
                results_json[exp_name][sname] = {'error': str(e)}
            sys.stdout.flush()

    # Summary
    print(f'\n{"="*80}', flush=True)
    print('SUMMARY', flush=True)
    print(f'{"="*80}', flush=True)
    print(f'{"System":<12} {"Sampler":<8} {"KS":>8} {"tau_E":>8} {"ESS/eval":>10} {"Wall(s)":>8}', flush=True)
    print('-'*60, flush=True)
    for exp_name in S.EXPERIMENTS:
        for sname in S.SAMPLER_NAMES:
            r = results_json[exp_name].get(sname, {})
            if 'error' in r:
                print(f'{exp_name:<12} {sname:<8} ERROR', flush=True)
                continue
            print(f'{exp_name:<12} {sname:<8} {r.get("energy_ks",0):>8.4f} '
                  f'{r.get("autocorr_time_energy",0):>8.1f} '
                  f'{r.get("ess_per_force_eval",0):>10.6f} '
                  f'{r.get("wall_time",0):>8.1f}', flush=True)

    # System-specific
    print('\nGAUSS 20D Variance Errors:', flush=True)
    for sname in S.SAMPLER_NAMES:
        r = results_json.get('Gauss_20D', {}).get(sname, {})
        if 'var_max_rel_error' in r:
            print(f'  {sname}: max={r["var_max_rel_error"]:.4f} mean={r["var_mean_rel_error"]:.4f}', flush=True)

    print('\nGMM 10D Mode Visitation:', flush=True)
    for sname in S.SAMPLER_NAMES:
        r = results_json.get('GMM_10D', {}).get(sname, {})
        if 'mode_transitions' in r:
            print(f'  {sname}: transitions={r["mode_transitions"]} visits={r["visits_per_mode"]} rate={r["transition_rate"]:.6f}', flush=True)

    # Save
    out_path = os.path.join(_this_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f'\nResults: {out_path}', flush=True)

    S.make_all_figures(results_plots, os.path.join(_this_dir, 'figures'))


if __name__ == '__main__':
    main()
