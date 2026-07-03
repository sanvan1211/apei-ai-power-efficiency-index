"""Model loading and single/batched generation for the APEI validation experiment.

Llama 4 Maverick doesn't fit on a free T4, so Qwen2.5-3B-Instruct stands in as
a same-class proxy to validate the method only. 3B in fp16 is ~6 GB and leaves
room for the KV cache on a 16 GB T4; an 8B model would not. Qwen2.5-3B is
ungated. Swap in a gated Llama repo (and authenticate) if you have access; the
method is identical.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen2.5-3B-Instruct'
MAX_NEW_TOKENS = 128

# A dozen varied prompts, greedy decoding, fixed max_new_tokens so runs are
# deterministic and comparable.
PROMPTS = [
    "Explain photosynthesis in two sentences.",
    "Write a haiku about a thunderstorm.",
    "List three causes of the French Revolution.",
    "What is the difference between TCP and UDP?",
    "Translate 'good morning, how are you?' into French.",
    "Give a recipe for a simple omelette.",
    "Summarize the plot of Romeo and Juliet briefly.",
    "What are the first five prime numbers?",
    "Describe how a hash map works.",
    "Write a short motivational quote about persistence.",
    "Explain the greenhouse effect to a 10-year-old.",
    "Name three uses of regular expressions in programming.",
]


def load_model():
    """Load the tokenizer and model onto the GPU. Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map='cuda',
    )
    model.eval()

    used_gb = torch.cuda.memory_allocated(0) / 1e9
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'Loaded {MODEL_ID}: {used_gb:.1f} GB / {total_mem_gb:.1f} GB in use.')
    return model, tokenizer


def prepare_tokenizer_for_batching(tokenizer):
    """Decoder-only models must left-pad so every sequence's real tokens end at
    the same right edge and generation continues from there."""
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def generate_one(model, tokenizer, energy_meter, prompt):
    """Run one chat-formatted prompt. Returns (output_tokens, energy_j, time_s)."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=False,  # bare tensor, not a dict
        return_tensors="pt",
    ).to("cuda")
    prompt_len = inputs.shape[1]

    with energy_meter.measure("gen") as meas:
        with torch.no_grad():
            out = model.generate(
                inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()  # let the energy window close on finished work

    new_tokens = out.shape[1] - prompt_len
    return new_tokens, meas["energy_j"], meas["time_s"]


def generate_batch(model, tokenizer, energy_meter, prompts):
    """Run a list of prompts as one padded batch. Returns (real_tokens, energy_j, time_s)."""
    texts = [
        tokenizer.apply_chat_template(
            [{'role': 'user', 'content': p}],
            add_generation_prompt=True, tokenize=False,
        )
        for p in prompts
    ]
    enc = tokenizer(texts, return_tensors='pt', padding=True).to('cuda')
    prompt_width = enc['input_ids'].shape[1]

    with energy_meter.measure('batch') as meas:
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        torch.cuda.synchronize()

    # Count only non-pad generated tokens. Sequences that hit EOS early are
    # pad-filled to the longest sequence; those pads aren't real output. (When
    # pad == eos this undercounts by ~1 token per finished sequence, negligible
    # against 128.)
    gen = out[:, prompt_width:]
    real_tokens = (gen != tokenizer.pad_token_id).sum().item()
    return real_tokens, meas['energy_j'], meas['time_s']
