"""Shim to import parent orbit's run_experiment under a clean module name."""
import importlib.util
import os

_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "q-optimization-035", "run_experiment.py"))
_spec = importlib.util.spec_from_file_location("q_opt_035_re", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

simulate_multiscale = _mod.simulate_multiscale
simulate_nhc = _mod.simulate_nhc
tau_q2_mean = _mod.tau_q2_mean
autocorr_time = _mod.autocorr_time
AnisotropicGaussian = _mod.AnisotropicGaussian
