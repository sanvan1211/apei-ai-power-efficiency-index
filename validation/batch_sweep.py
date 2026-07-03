"""Batch-size sensitivity sweep.

Characterizes how per-token energy falls as batch size grows (not a
validation of the TDP formula). Per-token energy falls sharply with batch
size because a large fixed baseline power (memory, clocks) is amortized over
more tokens. Production serving is high-batch, but the closed-model
throughput source doesn't disclose its batch size. If those figures reflect
low-batch measurement, the derived J/token (and gCO2e/token) is too high -
i.e. the dataset likely overstates real production per-token carbon, which is
the safe direction for a carbon-accounting claim.
"""

import gc
import math

import matplotlib.pyplot as plt
import pandas as pd
import torch

from .model_runner import PROMPTS, generate_batch, prepare_tokenizer_for_batching

BATCH_SIZES = [1, 4, 8, 16]  # 16 may OOM on a free T4; handled below


def make_batches(prompts, B):
    """Cycle the prompt list up to the next multiple of B so every batch is full."""
    n = math.ceil(len(prompts) / B) * B
    cycled = [prompts[i % len(prompts)] for i in range(n)]
    return [cycled[i:i + B] for i in range(0, n, B)]


def run(model, tokenizer, energy_meter):
    """Run the batch-size sweep, save results + chart, and return the sweep dataframe."""
    prepare_tokenizer_for_batching(tokenizer)

    _ = generate_batch(model, tokenizer, energy_meter, PROMPTS[:2])  # warm up the batched path

    sweep_rows = []
    for B in BATCH_SIZES:
        try:
            torch.cuda.empty_cache()
            tot_tok, tot_e, tot_t = 0, 0.0, 0.0
            for batch in make_batches(PROMPTS, B):
                tok, e, t = generate_batch(model, tokenizer, energy_meter, batch)
                tot_tok += tok
                tot_e += e
                tot_t += t
            sweep_rows.append({
                'batch_size': B,
                'total_tokens': tot_tok,
                'time_s': round(tot_t, 3),
                'energy_j': round(tot_e, 2),
                'j_per_token': round(tot_e / tot_tok, 4),
                'tokens_per_sec': round(tot_tok / tot_t, 2),
            })
            print(f'batch {B:>2}: {tot_tok} tok | {tot_e:6.1f} J | {tot_t:5.2f} s '
                  f'| {tot_e / tot_tok:.4f} J/tok | {tot_tok / tot_t:6.1f} tok/s')
        except RuntimeError as ex:
            if 'out of memory' in str(ex).lower():
                print(f'batch size {B} OOM on this T4, skipping.')
                gc.collect()
                torch.cuda.empty_cache()
                continue
            raise

    sweep_df = pd.DataFrame(sweep_rows)
    print('Completed batch sizes:', list(sweep_df['batch_size']))

    # Each batch size's J/token relative to batch 1 (the single-request baseline).
    base = sweep_df.loc[sweep_df['batch_size'] == 1, 'j_per_token'].iloc[0]
    sweep_df['pct_of_batch1'] = (100 * sweep_df['j_per_token'] / base).round(1)
    sweep_df['speedup_vs_b1'] = (base / sweep_df['j_per_token']).round(2)

    print(sweep_df.to_string(index=False))
    print()
    for _, r in sweep_df.iterrows():
        print(f"batch {int(r['batch_size']):>2}: {r['j_per_token']:.4f} J/tok  "
              f"= {r['pct_of_batch1']:.0f}% of batch-1 energy per token  "
              f"({r['speedup_vs_b1']:.2f}x cheaper)")

    _plot_sweep(sweep_df)

    sweep_csv = 'apei_batch_sweep_qwen_t4.csv'
    sweep_df.to_csv(sweep_csv, index=False)
    print('Saved', sweep_csv)

    try:
        from google.colab import files
        files.download(sweep_csv)
        files.download('apei_batch_sweep_chart.png')
    except Exception:
        pass

    return sweep_df


def _plot_sweep(sweep_df):
    """Chart: J/token vs batch size."""
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df['batch_size'], sweep_df['j_per_token'],
            marker='o', linewidth=2.5, markersize=11, color='#1f77b4')

    for _, r in sweep_df.iterrows():
        ax.annotate(f"{r['j_per_token']:.2f}",
                    (r['batch_size'], r['j_per_token']),
                    textcoords='offset points', xytext=(0, 13),
                    ha='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Batch size (concurrent requests)', fontsize=15)
    ax.set_ylabel('Energy per token  (J / token)', fontsize=15)
    ax.set_title('Per-token energy falls as batch size grows\nQwen2.5-3B-Instruct on Tesla T4',
                 fontsize=15)
    ax.set_xticks(sweep_df['batch_size'])
    ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)  # y-axis from zero
    fig.tight_layout()
    fig.savefig('apei_batch_sweep_chart.png', dpi=150, bbox_inches='tight')
    plt.show()
