import tqdm
import numpy as np

from vllm import LLM
from vllm.sampling_params import SamplingParams

def load_model(model_name:str):
    if model_name == 'mistralai/Pixtral-12B-2409':
        sampling_params = SamplingParams(max_tokens=8192, temperature=0.1)

        llm = LLM(model=model_name, tokenizer_mode="mistral", limit_mm_per_prompt={"image": 2}, max_model_len=32768)
        
        return llm, sampling_params

def pixtral_inference(
    llm, sampling_params, image_url_1, image_url_2, prompt:str=''
):
    image_pair_text = []
    for idx, (url_1, url_2) in tqdm.tqdm(enumerate(zip(image_url_1, image_url_2))):
        # Construct Prompt
        contents = [
            {
                "type" : "text",
                "text" : prompt
            },
        ]

        image_url_format = {
            "type" : "image_url"
        }
        
        # Format image url dict
        image_url_format_1 = image_url_format.copy()
        image_url_format_1["image_url"] = {"url" : url_1}
        
        image_url_format_2 = image_url_format.copy()
        image_url_format_2["image_url"] = {"url" : url_2}
        
        contents.append(image_url_format_1)
        contents.append(image_url_format_2)
        
        messages = [
            {
                "role": "user",
                "content": contents
            }
        ]
        
        outputs = llm.chat(
            messages, 
            sampling_params=sampling_params,
            #disable_log_stats=True,
            use_tqdm=False
        )[0].outputs[0].text
        
        image_pair_text.append(outputs)
        
    return np.array(image_pair_text)
