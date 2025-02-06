from gptqmodel import GPTQModel
import ipdb
model = GPTQModel.load("ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v2.5")

qq =model.model.model.layers[0].self_attn.q_proj

ipdb.set_trace()
result = model.generate("Uncovering deep insights begins with")[0]
print(result)