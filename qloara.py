from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
import torch
import ipdb
import bitsandbytes
bitsandbytes.nn.modules.Linear4bit
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
device='cuda'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map='cuda')



QQ = model.model.layers[0].self_attn.q_proj



def moderate(messages):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

print(moderate(messages))