from RegionalDiffusion_xl import RegionalDiffusionXLPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from mllm import GPT5
import torch
from huggingface_hub import login

###  PLEASE DON'T PUBLIC MY TOKEN  ###
hf_token = "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"
gpt_token = "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"
hf_cache_dir = "/workspace/my_models/cache"

login(token=hf_token)

pipe1 = RegionalDiffusionXLPipeline.from_pretrained("comin/IterComp", cache_dir=hf_cache_dir, torch_dtype=torch.float16, use_safetensors=True)
pipe1.to("cuda")
pipe1.scheduler = DPMSolverMultistepScheduler.from_config(pipe1.scheduler.config,use_karras_sigmas=True)
pipe1.enable_xformers_memory_efficient_attention()

## User input
prompt= 'A floating city above the clouds, with golden towers and waterfalls cascading into the mist below. A dragon with shimmering wings soars through the sky, while airships dock at crystal platforms.'
para_dict = GPT5(prompt, gpt_token)

split_ratio = para_dict['Final split ratio']
regional_prompt = para_dict['Regional Prompt']
split_ratio = para_dict['Final split ratio']
prompt_en = para_dict['Prompt En']

negative_prompt = ""

images = pipe1(
    prompt = regional_prompt,
    split_ratio = split_ratio,
    batch_size = 1, #batch size
    base_ratio = 0.5, # The ratio of the base prompt    
    base_prompt= prompt,       
    num_inference_steps=50, # sampling step
    height = 512, 
    negative_prompt=negative_prompt, # negative prompt
    width = 512, 
    seed = 2468, # random seed
    guidance_scale = 7.0
).images[0]

images.save("test.png")
print('done')