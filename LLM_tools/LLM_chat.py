import json
import requests
from transformers import set_seed
import os
import openai
from openai import AzureOpenAI

set_seed(47)


def vicuna_gen(prompt: str, url="http://192.168.0.90:5006"):
    vicuna_url = url
    headers = {"Content-Type": "application/json"}
    input_datas = {"prompt": prompt}
    input_datas = json.dumps(input_datas)
    response = requests.post(vicuna_url, headers=headers, data=input_datas)
    return response.text


def gpt3_chat(prompt: str, url=None):
    api_key = os.environ['api_key']
    azure_endpoint = os.environ['azure_endpoint']
    api_version = os.environ['api_version']
    client = AzureOpenAI(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)
    user_content = {"role": "user", "content": prompt}
    message_text = [{"role": "system", "content": "You are an AI assistant that helps people find information."},
                    user_content]
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo-1106',
        messages=message_text,
        seed=47,
        max_tokens=2048
    )
    return completion.choices[0].message.content


def openchat_3_5(user_prompt: str, url='http://192.168.0.90:18888/v1/chat/completions'):
    headers = {"Content-Type": "application/json"}
    datas = {"model": "openchat_3.5", "messages": [{'role': 'user', 'content': user_prompt}], "temperature": 0}
    json_data = json.dumps(datas)
    response = requests.post(url, headers=headers, data=json_data)
    return response.json()['choices'][0]['message']['content']


def mistral_7b(user_prompt, url="http://192.168.0.90:5008/"):
    headers = {"Content-Type": "application/json"}
    datas = {"prompt": user_prompt, "temperature": 0}
    json_data = json.dumps(datas)
    response = requests.post(url, headers=headers, data=json_data).text
    return response


def llm_chat(model_name, prompt, url):
    if model_name == 'openchat':
        return openchat_3_5(prompt, url)
    if model_name == 'vicuna':
        return vicuna_gen(prompt, url)
    if model_name == 'chatgpt':
        return gpt3_chat(prompt)
    if model_name == 'mistral':
        return mistral_7b(prompt, url)


class Sentence_Embedding:
    def __init__(self, url: str):
        self.url = url

    def text_embedding(self, text_list):
        headers = {"Content-Type": "application/json"}
        datas = {"text_list": text_list}
        json_data = json.dumps(datas)
        response = requests.post(self.url, headers=headers, data=json_data)
        return response.json()


if __name__ == '__main__':
    # temp = 'Generate no more than 10 noun keywords from the following text: [DOCUMENT]. Each noun key ' \
    #        'word consists of 1~4 words. ' \
    #        'Note that your answer is a strict and complete json (ended with a "}"), ' \
    #        'with a only key named "keywords", such as {"keywords":["a","b","c"]}.'
    #
    # file_path = '/mnt/c2251223-a6be-4a70-9d06-720c65eccb98/daishengran/deploy/Fewshot-Grounded-Keyphrase-Generator
    # /Datasets/nus/word_nus_testing_context.txt'
    #
    # with open(file_path, 'r') as f:
    #     for line in f:
    #         document = line.replace('<eos>', '')
    #
    #         res = vicuna_gen([temp.replace('DOCUMENT', document)])
    #         # res = chinese_llama2_gen(temp.replace('DOCUMENT', document))
    #         # res = chinese_llama2_gen("the color pf banana is")
    #         print(res)
    pass
    sss = 'Here is a provided target document: on the emergence of rules in neural networks . <eos> grammars ) from ' \
          'sequences of arbitrary input symbols by inventing abstract representations that accommodate unseen symbol ' \
          'sets as well as unseen but similar grammars . the neural network is shown to have the ability to transfer ' \
          'grammatical knowledge to both new symbol vocabularies and new grammars . analysis of the state space shows ' \
          'that the network learns generalized abstract structures of the input and is not simply memorizing the ' \
          'input strings . these representations are context sensitive , hierarchical , and based on the state ' \
          'variable of the finite state machines that the neural network has learned . generalization to new symbol ' \
          'sets or grammars arises from the spatial nature of the internal representations used by the network , ' \
          'allowing new symbol sets to be encoded close to symbol sets that have already been learned in the hidden ' \
          'unit space of the network . the results are counter to the arguments that learning algorithms based on ' \
          'weight adaptation after each exemplar presentation ( such as the long term potentiation found in the ' \
          'mammalian nervous system ) can not in principle extract symbolic knowledge from positive examples as ' \
          'prescribed by prevailing human linguistic theory and evolutionary psychology. Here are two provided ' \
          'phrases:\'artificial intelligence\' and \'symbol grammars\'.The question is: which phrase above is more ' \
          'suitable to be the generalized hypernym phrase of the above target document? Note that you should select ' \
          'one phrase from above two provided phrases. Please return your analysis process and put your final answer ' \
          'in a json format, with a key named \'answer\'.-> Due to the document mentioned \'neural networks\' for ' \
          'several times, and \'artificial intelligence\' is the hypernym word of \'neural networks\', so the answer ' \
          'is: {"answer": "artificial intelligence"}\nHere is a provided target document: power electronics spark new ' \
          'simulation challenges . <eos> systems and explores some of the inherent requirements for simulation ' \
          'technologies in order to keep up with this rapidly changing environment . the authors describe how energy ' \
          'utilities are realizing that , with the appropriate tools , they can train and sustain engineers who can ' \
          'maintain a great insight into system dynamics. Here are two provided phrases: \'electric utilities\' and ' \
          '\'AI system\'. The question is: which phrase above is more suitable to be the generalized hypernym phrase ' \
          'of the above target document? Note that you should select one phrase from above two provided phrases. ' \
          'Please return your analysis process and put your final answer in a json format, with a key named ' \
          '\'answer\'.->'
    ss = 'Target document:\npower electronics spark new ' \
         'simulation challenges . <eos> systems and explores some of the inherent requirements for simulation ' \
         'technologies in order to keep up with this rapidly changing environment . the authors describe how energy ' \
         'utilities are realizing that , with the appropriate tools , they can train and sustain engineers who can ' \
         'maintain a great insight into system dynamics.\n Note that you should put your answer in a json format ' \
         'with only one key named \'key phrases\' ,and each keyword consists of 1~4 words.'
    # system_prompt = 'generate at least 5 key phrases from the given target document'
    # messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': ss}]
    # res = sentence_transformers(['hello'], url='http://192.168.0.237:6666')
    # print(res)
    prompt_temp = "the real meaning of life is"

    res = llm_chat('chatgpt', prompt_temp, url='http://192.168.0.90:5009')
    print(res)
