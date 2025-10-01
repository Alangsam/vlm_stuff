import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Efficient-Large-Model/VILA1.5-3b"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True
).to(device)

messages = [{"role": "user", "content": "In one sentence, explain why the sky appears blue."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)

with torch.inference_mode():
    out = model.generate(inputs, max_new_tokens=128, do_sample=False)

print(tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True))
