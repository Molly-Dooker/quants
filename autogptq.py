
import auto_gptq.nn_modules
import auto_gptq.nn_modules.qlinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda_old
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LlamaForCausalLM
import ipdb
import auto_gptq
auto_gptq.nn_modules.qlinear.qlinear_cuda_old.QuantLinear
model_id = "iqbalamo93/Meta-Llama-3.1-8B-Instruct-GPTQ-Q_8"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)
tokenizer = AutoTokenizer.from_pretrained(model_id) 


# qq = model.model.layers[0].self_attn.q_proj

# ipdb.set_trace()

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline(
    "text-generation",
    model=model,
    device_map="auto",
    tokenizer=tokenizer,
)

generation_args = { 
    "max_new_tokens": 500, 
    "return_full_text": False, 
    "temperature": 0.1, 
    "do_sample": False,
    "pad_token_id": 128001 
} 

output = pipe(messages, **generation_args) 
print(output[0]['generated_text'])
