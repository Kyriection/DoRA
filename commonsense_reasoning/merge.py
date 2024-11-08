import numpy 
import os 
import sys 
import torch 
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.join(os.getcwd(), "peft/src/"))
from peft import PeftModel

lora_weights = sys.argv[1]

model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True)

model = PeftModel.from_pretrained(
        model,
        lora_weights,
        torch_dtype=torch.float16,
        device_map={"":0})

checkpoints = {}
for name, module in model.named_modules():
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        print('Merging', name)
        lora_weight = module.lora_B.weight @ module.lora_A.weight * module.scaling
        checkpoints[name+'.weight'] = module.weight.data + lora_weight


for name, p in model.named_parameters():
    if not name in checkpoints:
        if not 'lora' in name:
            checkpoints[name] = p.data

print(checkpoints.keys())

torch.save(checkpoints, "llama2_7b_ft_50sparsity/pytorch_model.bin")
