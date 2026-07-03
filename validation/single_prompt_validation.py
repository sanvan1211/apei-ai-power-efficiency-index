"""Single-prompt validation run.

Measures per-prompt J/token on real hardware and compares the aggregate
against the TDP-derived prediction used for the closed models in APEI:

    NodePower = chips_per_node * chip_TDP * overhead
    predicted J/token = NodePower / throughput

Measured throughput is plugged in, so only the power assumption is under
test. Single-GPU case: chips_per_node = 1, chip_TDP = 70 W (T4).

Interpretation: if predicted J/token lands within ~15% of measured, the
TDP-x-1.4 recipe reproduces ground truth and the same recipe on TPU v7 /
Trainium2 is defensible; if not, report the implied overhead constant
instead. Caveat: a single 70 W T4 is a weak analog for a multi-chip node, so
this validates the arithmetic and the TDP-to-draw ratio, not the node
multiplier itself.
"""

import pandas as pd

from .energy_backend import T4_TDP_WATTS
from .model_runner import MODEL_ID, PROMPTS, generate_one

CHIPS_PER_NODE = 1
OVERHEAD = 1.4


def run(model, tokenizer, energy_meter, gpu_name):
    """Run the full single-prompt validation and save results to CSV."""
    _ = generate_one(model, tokenizer, energy_meter, "Say hello.")  # warm-up, not recorded

    rows = []
    for i, p in enumerate(PROMPTS):
        tok, energy_j, t_s = generate_one(model, tokenizer, energy_meter, p)
        rows.append({
            'idx': i,
            'prompt': p[:40] + ('...' if len(p) > 40 else ''),
            'out_tokens': tok,
            'time_s': round(t_s, 3),
            'energy_j': round(energy_j, 2),
            'throughput_tok_s': round(tok / t_s, 2),
            'j_per_token': round(energy_j / tok, 4),
        })
        print(f'[{i + 1}/{len(PROMPTS)}] {tok} tok | {energy_j:.1f} J | {t_s:.2f} s')

    df = pd.DataFrame(rows)
    print(df)

    total_energy = df['energy_j'].sum()
    total_tokens = int(df['out_tokens'].sum())
    total_time = df['time_s'].sum()

    measured_j_per_token = total_energy / total_tokens
    agg_throughput = total_tokens / total_time  # tok/s across the whole batch

    node_power = CHIPS_PER_NODE * T4_TDP_WATTS * OVERHEAD
    predicted_j_per_token = node_power / agg_throughput

    pct_diff = 100.0 * (predicted_j_per_token - measured_j_per_token) / measured_j_per_token
    implied_overhead = OVERHEAD * measured_j_per_token / predicted_j_per_token
    measured_avg_power = total_energy / total_time

    print(f'Total: {total_tokens} tokens, {total_energy:.1f} J, {total_time:.1f} s')
    print(f'Aggregate throughput      : {agg_throughput:.2f} tok/s')
    print(f'Measured avg GPU power     : {measured_avg_power:.1f} W  '
          f'(vs {T4_TDP_WATTS} W TDP, x{OVERHEAD} = {node_power:.0f} W assumed)')
    print('-' * 50)
    print(f'MEASURED  J/token : {measured_j_per_token:.4f}')
    print(f'PREDICTED J/token : {predicted_j_per_token:.4f}')
    print(f'% difference      : {pct_diff:+.1f}%  (predicted vs measured)')
    print(f'Implied overhead constant that fits: {implied_overhead:.2f}  (you assumed {OVERHEAD})')

    # Save per-prompt rows plus an aggregate summary row to one CSV.
    summary = {
        'idx': 'AGGREGATE', 'prompt': f'{MODEL_ID} on {gpu_name}',
        'out_tokens': total_tokens,
        'time_s': round(total_time, 3),
        'energy_j': round(total_energy, 2),
        'throughput_tok_s': round(agg_throughput, 2),
        'j_per_token': round(measured_j_per_token, 4),
        'predicted_j_per_token': round(predicted_j_per_token, 4),
        'pct_diff': round(pct_diff, 1),
        'implied_overhead': round(implied_overhead, 2),
        'backend': energy_meter.backend,
    }
    out_df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    csv_path = 'apei_validation_qwen25_3b_t4.csv'
    out_df.to_csv(csv_path, index=False)
    print('Saved', csv_path)

    try:
        from google.colab import files
        files.download(csv_path)
    except Exception:
        pass

    return df, out_df
