# file: quick_vila15_3b_text.py
# minimal text-only prompt to VILA-1.5-3B with pure PyTorch/Transformers

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Efficient-Large-Model/VILA1.5-3b"

# 1) Device + dtype: use CUDA if present, else CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# fp16 on CUDA saves memory; fall back to float32 on CPU
dtype = torch.float16 if device == "cuda" else torch.float32

# 2) Load tokenizer and model
# VILA uses a custom modeling code path (trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device)

# 3) Build a chat-style prompt via the tokenizer's chat template
# Keep it simple: one user message
messages = [
    {"role": "user", "content": "In one sentence, explain why the sky appears blue."}
]

# apply_chat_template formats the conversation the way the model expects
inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(device)

# 4) Generate
with torch.inference_mode():
    output_ids = model.generate(
        inputs,
        max_new_tokens=128,
        do_sample=False  # greedy for determinism; flip to True and add temperature for sampling
    )

# 5) Decode only the new tokens
generated_text = tok.decode(output_ids[0][inputs.shape[-1]:], skip_special_tokens=True)
print(generated_text)
