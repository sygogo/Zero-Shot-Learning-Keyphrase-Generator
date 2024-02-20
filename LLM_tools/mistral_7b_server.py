import json
import os
import torch
from flask import Flask, request
from vllm import LLM, SamplingParams

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def get_datas():
    re_data = request.get_json()
    print(re_data)
    prompt = re_data.get('prompt')
    max_new_tokens = re_data.get('max_new_tokens', 4096)
    top_k = re_data.get('top_k', 150)
    top_p = re_data.get('top_p', 0.9)
    temperature = re_data.get('temperature', 0.6)
    repetition_penalty = re_data.get('repetition_penalty', 1.05)
    sample_params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, top_p=top_p, top_k=top_k,
                                   repetition_penalty=repetition_penalty)
    answer = model.generate(prompt, sampling_params=sample_params)[0].outputs[0].text
    torch.cuda.empty_cache()
    return answer


def init_model(model_path: str, seed=47):
    llm_model = LLM(model_path, seed=seed, enforce_eager=True)
    return llm_model


if __name__ == '__main__':
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    # model_path = '/mnt/c2251223-a6be-4a70-9d06-720c65eccb98/daishengran/deploy/Models/Mistral_7B_Instruct_v0.2'
    model = init_model(config['mistral_path'])
    app.run(host=config['mistral_ip'], port=config['mistral_port'], debug=False, threaded=True)
