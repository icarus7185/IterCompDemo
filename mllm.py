import requests
import json
import os
import re
# from transformers import AutoTokenizer
# import transformers
# import torch
# from transformers import LlamaForCausalLM, LlamaTokenizer

def GPT5(prompt, api_token):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = api_token
    with open('template/template2.txt', 'r') as f:
        template=f.readlines()
    user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
    
    textprompt= f"{' '.join(template)} \n {user_textprompt}"
    
    payload = json.dumps({
    "model": "gpt-5.2", # we suggest to use the latest version of GPT, you can also use gpt-4-vision-preivew, see https://platform.openai.com/docs/models/ for details. 
    "messages": [
        {
            "role": "user",
            "content": textprompt
        }
    ]
    })
    headers = {
    'Accept': 'application/json',
    'Authorization': f'Bearer {api_key}',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json'
    }
    print('waiting for GPT-5 response')
    response = requests.request("POST", url, headers=headers, data=payload)
    obj=response.json()
    text=obj['choices'][0]['message']['content']
    print('------response--begin------')
    print(text)
    print('------response--end------')
    # Extract the split ratio and regional prompt

    return get_params_dict(text, prompt)

# def local_llm(prompt,version,model_path=None):
#     if model_path==None:
#         model_id = "Llama-2-13b-chat-hf" 
#     else:
#         model_id=model_path
#     print('Using model:',model_id)
#     tokenizer = LlamaTokenizer.from_pretrained(model_id)
#     model = LlamaForCausalLM.from_pretrained(model_id, load_in_8bit=False, device_map='auto', torch_dtype=torch.float16)
#     with open('template/template.txt', 'r') as f:
#         template=f.readlines()
#     user_textprompt=f"Caption:{prompt} \n Let's think step by step:"
#     textprompt= f"{' '.join(template)} \n {user_textprompt}"
#     model_input = tokenizer(textprompt, return_tensors="pt").to("cuda")
#     model.eval()
#     with torch.no_grad():
#         print('waiting for LLM response')
#         res = model.generate(**model_input, max_new_tokens=1024)[0]
#         output=tokenizer.decode(res, skip_special_tokens=True)
#         output = output.replace(textprompt,'')
#     return get_params_dict(output)

def get_params_dict(output_text, original):
    response = output_text
    ratio = True
    prompt = True
    # Find Final split ratio
    split_ratio_match = re.search(r"Final split ratio: ([\d.,;]+)", response)
    if split_ratio_match:
        final_split_ratio = split_ratio_match.group(1)
        print("Final split ratio:", final_split_ratio)
    else:
        print("Final split ratio not found.")

    # Find Regional Prompt
    prompt_match = re.search(r"Regional Prompt: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_match:
        regional_prompt = prompt_match.group(1).strip()
        print("Regional Prompt:", regional_prompt)
    else:
        prompt_match2 = re.search(r"Region0 \(Row.*,width=.*\)*: \(Row.*,width=.*\)* (.*?)(?=\n\n|\Z)", response, re.DOTALL)
        print("Regional Prompt not found")
        
    prompt_en_match = re.search(r"CaptionEn: (.*?)(?=\n\n|\Z)", response, re.DOTALL)
    if prompt_en_match:
        prompt_en = prompt_en_match.group(1).strip()
        print("Prompt En:", prompt_en)
    else:
        prompt_en = original

    image_region_dict = {'Final split ratio': final_split_ratio, 'Regional Prompt': regional_prompt, 'Prompt En': prompt_en}    
    return image_region_dict
