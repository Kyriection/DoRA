import numpy 
import os 
import torch 
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
        device_map={"":0}
    )

import pdb; pdb.set_trace()
