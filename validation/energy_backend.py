"""GPU check and energy-measurement backend for the APEI validation experiment.

APEI (gCO2e/token) derives energy for the closed APIs from published chip TDP
specs, which we can't measure directly. This module measures real GPU energy
so it can be checked against the same TDP-derived formula.

Tries Zeus first, falls back to polling NVML directly. Both read
nvmlDeviceGetTotalEnergyConsumption (cumulative GPU energy, mJ), supported on
Volta and newer - the T4 is Turing, so it qualifies.
"""

import contextlib
import threading
import time

import torch

# Published Tesla T4 TDP - the "published spec" input the closed-model formula uses.
T4_TDP_WATTS = 70.0


def check_gpu():
    """Verify a CUDA GPU is available and print its name/memory.

    Returns (name, total_mem_gb).
    """
    assert torch.cuda.is_available(), 'No GPU. Set Runtime -> Change runtime type -> T4 GPU.'

    name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {name} ({total_mem_gb:.1f} GB)')

    if 'T4' not in name:
        print('Warning: not a T4, so the 70 W TDP assumption below will not apply.')

    return name, total_mem_gb


class EnergyMeter:
    """Measures GPU energy for a block of code, via Zeus or NVML fallback."""

    def __init__(self):
        self.backend, self._zeus_monitor, self._handle = self._setup_backend()

    @staticmethod
    def _setup_backend():
        """Detect which energy-measurement backend is available."""
        backend = None
        zeus_monitor = None
        handle = None

        try:
            from zeus.monitor import ZeusMonitor
            zeus_monitor = ZeusMonitor(gpu_indices=[0])
            # smoke test: a tiny window must return a real number
            zeus_monitor.begin_window('smoke')
            time.sleep(0.2)
            m = zeus_monitor.end_window('smoke')
            assert m.total_energy is not None and m.total_energy >= 0
            backend = 'zeus'
        except Exception as e:
            print(f'Zeus unavailable ({type(e).__name__}: {e}); falling back to pynvml.')

        if backend is None:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # raises if unsupported
                backend = 'nvml_energy'
            except Exception:
                backend = 'nvml_power'  # no energy counter; integrate power draw instead

        print('Energy backend:', backend)
        return backend, zeus_monitor, handle

    @contextlib.contextmanager
    def measure(self, label='gen'):
        """Yield a dict that, on exit, holds 'energy_j' and 'time_s' for the enclosed block."""
        out = {}
        t0 = time.perf_counter()

        if self.backend == 'zeus':
            self._zeus_monitor.begin_window(label)
            try:
                yield out
            finally:
                m = self._zeus_monitor.end_window(label)
                out['energy_j'] = float(m.total_energy)
                out['time_s'] = time.perf_counter() - t0

        elif self.backend == 'nvml_energy':
            import pynvml
            e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)  # mJ
            try:
                yield out
            finally:
                e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                out['energy_j'] = (e1 - e0) / 1000.0
                out['time_s'] = time.perf_counter() - t0

        else:  # nvml_power: sample power at 20 Hz and integrate
            import pynvml
            samples = []
            stop = threading.Event()

            def poll():
                while not stop.is_set():
                    w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0  # mW -> W
                    samples.append((time.perf_counter(), w))
                    time.sleep(0.05)

            th = threading.Thread(target=poll)
            th.start()
            try:
                yield out
            finally:
                stop.set()
                th.join()
                e = 0.0
                for (ta, wa), (tb, wb) in zip(samples, samples[1:]):
                    e += 0.5 * (wa + wb) * (tb - ta)  # trapezoidal W*s -> J
                out['energy_j'] = e
                out['time_s'] = time.perf_counter() - t0
