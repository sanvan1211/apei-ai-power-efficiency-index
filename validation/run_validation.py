"""
APEI methodology validation on Qwen2.5-3B-Instruct (T4 GPU).

Entry point for the full validation experiment: GPU/energy backend setup,
model load, single-prompt J/token measurement vs. the TDP-derived
prediction, and the batch-size sensitivity sweep.

Run on a T4 GPU runtime (e.g. a Google Colab T4 instance):

    pip install -q --upgrade transformers accelerate zeus-ml pynvml
    python -m validation.run_validation
"""

from . import batch_sweep, single_prompt_validation
from .energy_backend import EnergyMeter, check_gpu
from .model_runner import load_model


def main():
    gpu_name, _total_mem_gb = check_gpu()
    energy_meter = EnergyMeter()
    model, tokenizer = load_model()

    single_prompt_validation.run(model, tokenizer, energy_meter, gpu_name)
    batch_sweep.run(model, tokenizer, energy_meter)


if __name__ == '__main__':
    main()
