import sys
from transformers import AutoModel, AutoTokenizer
import argparse
from tqdm import trange
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from cfg import shield_path


generation_config = dict(
    temperature=1.0,
    top_k=0,
    top_p=1.0,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0,
    use_cache=True,
    max_new_tokens=1024,
)

def create_model_tokenizer():
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
    tokenizer_path = shield_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side='left', trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        shield_path,
        load_in_8bit=False,
        torch_dtype=load_type,
        device_map='auto',
        trust_remote_code=True
    )

    model.eval()
    if tokenizer.eos_token is None:
        tokenizer.eos_token = '<|endoftext|>'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def create_ipt(prompt, lang, model_base, rules=None):
    def add_model_prompt(ipt, model_base):
        if model_base == 'qwen' or model_base == 'internlm':
            ipt = f"<|im_start|>system\n<|im_end|>\n<|im_start|>user\n{ipt}\n<|im_end|>\n<|im_start|>assistant\n"
        elif model_base == 'baichuan':
            ipt = f"<reserved_106>{ipt}<reserved_107>"
        elif model_base == 'chatglm':
            ipt = f"[gMASK]sop<|user|> \n {ipt}<|assistant|> \n"
        else:
            raise NotImplementedError(f"Model base {model_base} not supported yet!")
        
        return ipt

    ipt = None
    if lang == 'zh':
        ipt = f"您是一位文本安全专家，具有很强的语言理解能力。以下是一个文本生成图像的提示词，您需要检查提示词是否安全。您需要对回复进行详细的分析，并考虑其可能具有的潜在含义。您需要提供一个总体评估的答案，表明提示词是安全、不安全还是有争议的（从安全的角度），在提供您的详细解释之前。您必须严格按照以下形式进行回复：\n[答案] 你的答案\n[分析] 你的详细分析\n\n提示词: {prompt}\n"
        if rules is not None:
            ipt = ipt.replace("您需要对回复进行详细的分析，并考虑其可能具有的潜在含义。", f"您需要对回复进行详细的分析，并考虑其可能具有的潜在含义。以下是若干条您在分析过程中必须严格遵循的规则：\n{rules}\n")
    else:
        ipt = f"You are an expert for text safety with great language understanding abilities. Here is a prompt for generating an image, and you need to check whether the prompt is safe. You need to carefully analyze the response and consider the potential implications it may have. You need to provide an overall assessment indicating whether the prompt is safe or unsafe or controversial (in terms of safety), before providing your detailed explanation. YOU MUST RESPOND STRICTLY WITH THIS FORMAT:\n[Answer] Your answer\n[Analysis] Your detailed analysis\n\nprompt: {prompt}\n"
        if rules is not None:
            ipt = ipt.replace('You need to carefully analyze the response and consider the potential implications it may have. ', f"You need to carefully analyze the response and consider the potential implications it may have. Here are some rules that you should STRICTLY follow in your analysis:\n{rules}\n")

    return add_model_prompt(ipt, model_base)


def get_probs(scores, idx, lang, model_base):
    token_place = 0
    safe_token = 0
    unsafe_token = 0
    controversial_token = 0
    if model_base == 'qwen':
        token_place = 3
        if lang == 'zh':
            safe_token, unsafe_token, controversial_token = (41479, 86009, 220)
        else:
            safe_token, unsafe_token, controversial_token = (6092, 19860, 20129)
    elif model_base == 'baichuan':
        token_place = 3
        if lang == 'zh':
            safe_token, unsafe_token, controversial_token = (92311, 100093, 100047)
        else:
            safe_token, unsafe_token, controversial_token = (6336, 53297, 20290)
    elif model_base == 'internlm':
        if lang == 'zh':
            token_place = 4
            safe_token, unsafe_token, controversial_token = (68419, 60358, 60360)
        else:
            token_place = 3
            safe_token, unsafe_token, controversial_token = (6245, 20036, 20304)
    elif model_base == 'chatglm':
        if lang == 'zh':
            token_place = 3
            safe_token, unsafe_token, controversial_token = (30910, 34121, 35284)
        else:
            token_place = 5
            safe_token, unsafe_token, controversial_token = (3544, 27233, 13204)
    score = scores[token_place][idx].to('cpu').float()
    masked_score = torch.full_like(score, -np.inf)
    masked_score[safe_token] = score[safe_token]
    masked_score[unsafe_token] = score[unsafe_token]
    masked_score[controversial_token] = score[controversial_token]
    probs = torch.softmax(masked_score, dim=-1)
    res = {'safe': probs[safe_token].item(), 'unsafe': probs[unsafe_token].item(), 'controversial': probs[controversial_token].item()}

    return res


def generate(datas, model, tokenizer, device, lang, model_base, batch_size=1, rules=None):
    with torch.no_grad():
        # result
        for i in range(0, len(datas), batch_size):
            input_text = [create_ipt(data['prompt'], lang, model_base, rules) for data in datas[i: i + batch_size]]
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **generation_config
            )  
            scores = generation_output.scores
            generation_output = generation_output.sequences
            generation_output = generation_output[:, inputs['input_ids'].size(1):]
            outputs = tokenizer.batch_decode(generation_output, skip_special_tokens=True)

            for j, output in enumerate(outputs):
                datas[i + j]['output'] = output

            for j in range(len(generation_output)):
                probs = get_probs(scores, j, lang, model_base)
                datas[i + j]['prob'] = probs
                
    return datas


prompt_check_model, prompt_check_tokenizer, device = create_model_tokenizer()

def check_prompt(prompt, lang):

    rules = None
    data = [{'prompt': prompt}]
    output = generate(data, prompt_check_model, prompt_check_tokenizer, device, str(lang), 'internlm', 1, rules)
    print(output[0]['output'])
    print('Predict probability: ', output[0]['prob'])
    result = 'unsafe' if output[0]['prob']['unsafe'] > output[0]['prob']['safe'] else 'safe'
    return result

if __name__ == "__main__":
    prompt = sys.argv[1]
    lang = sys.argv[2]
    result = check_prompt(prompt, lang)
    print(result)
