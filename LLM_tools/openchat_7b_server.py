import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request
from vllm import LLM, SamplingParams
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
app = Flask(__name__)


def init_llama2_model(path):
    model = AutoModelForCausalLM.from_pretrained(
        path, device_map='auto',
        trust_remote_code=False)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        use_fast=True)
    model = model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


@app.route('/', methods=['POST', 'GET'])
def get_datas():
    re_data = request.get_json()
    print(re_data)
    prompt = re_data.get('prompt')
    max_new_tokens = re_data.get('max_new_tokens', 512)
    top_k = re_data.get('top_k', 20)
    top_p = re_data.get('top_p', 0.9)
    temperature = re_data.get('temperature', 0.9)
    seed = re_data.get('seed', 42)
    set_seed(seed)
    answer = generate(prompt, max_new_tokens, top_k, top_p, temperature)
    torch.cuda.empty_cache()
    return answer


def generate(prompt: str,
             max_new_tokens=512,
             top_k=150,
             top_p=0.95,
             temperature=0.9):

    prompt_template = "GPT4 Correct User: {user}<|end_of_turn|>GPT4 Correct Assistant:"

    user_prompt = prompt_template.replace('{user}', prompt)

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens, top_k=top_k,
                                     use_beam_search=False)

    outputs = llm.generate([user_prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    return generated_text


if __name__ == '__main__':
    with open('../config.json','r') as config_file:
        config = json.load(config_file)
    llm = LLM(model=config['openchat_path'])
    app.run(host=config['openchat_ip'], port=config['openchat_port'], debug=False, threaded=True)
