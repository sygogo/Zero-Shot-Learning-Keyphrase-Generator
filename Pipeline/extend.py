import random
import tqdm
from utility import json_extract
from LLM_tools.LLM_chat import llm_chat
import urllib3

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from transformers import set_seed

set_seed(42)


class LLM_Extender:
    def __init__(self, extend_config: dict):
        if "extend_demos" in extend_config:
            self.extend_demos = extend_config['extend_demos']
        if "extend_prompt" in extend_config:
            self.extend_prompt = extend_config['extend_prompt']
        if "extend_mode" in extend_config:
            self.extend_mode = extend_config['extend_mode']
        if "LLM_url" in extend_config:
            self.LLM_url = extend_config['LLM_url']
        if "seed" in extend_config:
            self.seed = extend_config['seed']
        if "extend_num" in extend_config:
            self.extend_num = extend_config['extend_num']
        if "model" in extend_config:
            self.model = extend_config['model']
        if "extend_example_keywords" in extend_config:
            self.extend_example_keywords = extend_config['extend_example_keywords']
        if "extend_example_document" in extend_config:
            self.extend_example_document = extend_config['extend_example_document']

    def extend_phrases(self,
                       target_phrases: list,
                       target_document: str = None
                       ):

        if self.extend_mode == 'context_extend':
            extend_rounds = 1
            extended_phrases = []
            for index in tqdm.tqdm(range(extend_rounds), colour='green', desc='扩展', maxinterval=100):
                random.shuffle(target_phrases)
                present_keyphrases_group_str = '(' + ','.join(["'" + item + "'" for item in target_phrases]) + ')'
                user_prompt = self.extend_prompt.replace('<DOCUMENT>', target_document).replace('<CANDIDATE>',
                                                                                                present_keyphrases_group_str)
                response = llm_chat(self.model, prompt=user_prompt, url=self.LLM_url)
                try:
                    response_txt = response.replace('\n', '').strip(' ').strip('\t')
                    response_phrases_temp = json_extract(response_txt)
                    if response_phrases_temp:
                        extended_phrases.extend(response_phrases_temp)
                    else:
                        pass
                except:
                    pass
            print('extended_phrases: ', extended_phrases)
            return extended_phrases
