from utility import json_extract
from LLM_tools.LLM_chat import llm_chat
import urllib3

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLM_Extractor:
    def __init__(self, extract_config: dict):
        if "extract_prompt" in extract_config:
            self.extract_prompt = extract_config['extract_prompt']
        if "extract_num" in extract_config:
            self.extract_num = extract_config['extract_num']
        if "LLM_url" in extract_config:
            self.LLM_url = extract_config['LLM_url']
        if "seed" in extract_config:
            self.seed = extract_config['seed']
        if "model" in extract_config:
            self.model = extract_config['model']

    def extract_phrases(self, to_be_extracted_doc: str):
        response_phrases = []
        user_prompt = self.extract_prompt.replace('<DOCUMENT>', to_be_extracted_doc).replace('<NUM>',
                                                                                             str(self.extract_num))
        response = llm_chat(self.model, prompt=user_prompt, url=self.LLM_url)
        try:
            response_txt = response.replace('\n', '').strip(' ').strip('\t')
            response_phrases_temp = json_extract(response_txt)
            if response_phrases_temp:
                response_phrases.extend(response_phrases_temp)
                print("抽取：", response_phrases_temp)
            else:
                pass
        except:
            pass

        return response_phrases
