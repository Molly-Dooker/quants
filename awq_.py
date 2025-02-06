import awq.modules
import awq.modules.linear
import torch
import ipdb
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig, LlamaForCausalLM
import awq
awq.modules.linear.WQLinear_GEMM
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"

quantization_config = AwqConfig(
    bits=4,
    fuse_max_seq_len=512, # Note: Update this as per your use-case
    # do_fuse=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
  model_id,
  torch_dtype=torch.float16,
  low_cpu_mem_usage=True,
  device_map="cuda:1",
  quantization_config=quantization_config
)

prompt = [
  {"role": "system", "content": "You are a helpful assistant, that responds as a pirate."},
  {"role": "user", "content": "What's Deep Learning?"},
]
inputs = tokenizer.apply_chat_template(
  prompt,
  tokenize=True,
  add_generation_prompt=True,
  return_tensors="pt",
  return_dict=True,
).to("cuda:1")

qq  = model.model.layers[0].self_attn.q_proj
ipdb.set_trace()
outputs = model.generate(**inputs, do_sample=True, max_new_tokens=256)
print(tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0])