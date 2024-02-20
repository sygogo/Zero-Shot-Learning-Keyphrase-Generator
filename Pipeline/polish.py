from utility import phrase_stmmer, stem_extraction_
from LLM_tools.LLM_chat import llm_chat
import urllib3

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLM_Polisher:
    def __init__(self, polish_config: dict):
        if "polish_prompt" in polish_config:
            self.polish_prompt = polish_config['polish_prompt']
        if "extract_prompt" in polish_config:
            self.extract_prompt = polish_config['extract_prompt']
        if "LLM_url" in polish_config:
            self.LLM_url = polish_config['LLM_url']
        if "polish_rounds" in polish_config:
            self.polish_rounds = polish_config['polish_rounds']
        if "extractor" in polish_config:
            self.extractor = polish_config['extractor']
        if "model" in polish_config:
            self.model = polish_config['model']

    def polish_document(self, document):
        document_temp = document
        polish_phrases = []
        rounds = self.polish_rounds
        polish_documents = []
        while rounds > 0:
            user_prompt = self.polish_prompt.replace('<DOCUMENT>', document_temp)
            response = llm_chat(self.model, user_prompt,url=self.LLM_url)
            response_txt = response.replace('\n', '').strip(' ').strip('\t')
            if not response_txt:
                print('polish failed!')
                break
            if response_txt:
                polish_documents.append(response_txt)
                document_temp = response_txt
                extracted_phrases = self.extractor.extract_phrases(response_txt)
                if extracted_phrases:
                    polish_phrases.extend(extracted_phrases)

            rounds -= 1
        polish_phrases_stem = [stem_extraction_(item, phrase_stmmer) for item in polish_phrases]
        polish_phrases_stem_set = set(polish_phrases_stem)
        polish_phrases_freq = {polish_phrases[polish_phrases_stem.index(key)]: polish_phrases_stem.count(key) for key in polish_phrases_stem_set}
        polish_phrases_freq = dict(sorted(polish_phrases_freq.items(), key=lambda i: i[1], reverse=True))
        polish_phrases_freq = {key: value for key, value in polish_phrases_freq.items() if value > 1}
        return list(polish_phrases_freq.keys())[:20], polish_documents
