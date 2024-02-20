import collections
import math
import random
import tqdm
from utility import json_extract, remove_repeated
from LLM_tools.LLM_chat import openchat_3_5, mistral_7b, gpt3_chat, llm_chat
import urllib3

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from transformers import set_seed

set_seed(47)


class LLM_Ranker:
    def __init__(self, rank_config: dict):
        if "rank_demos" in rank_config:
            self.rank_demos = rank_config['rank_demos']
        if "rank_mode" in rank_config:
            self.rank_mode = rank_config['rank_mode']
        if "text_embedding" in rank_config:
            self.text_embedding = rank_config["text_embedding"]
        if "rank_prompt_positive" in rank_config:
            self.rank_prompt_positive = rank_config['rank_prompt_positive']
        if "rank_prompt_negative" in rank_config:
            self.rank_prompt_negative = rank_config["rank_prompt_negative"]
        if "rank_rounds" in rank_config:
            self.rank_rounds = rank_config['rank_rounds']
        if "LLM_url" in rank_config:
            self.LLM_url = rank_config['LLM_url']
        if "seed" in rank_config:
            self.seed = rank_config['seed']
        if "model" in rank_config:
            self.model = rank_config['model']
        if "reward_factor" in rank_config:
            self.reward_factor = rank_config['reward_factor']
        if "penalty_factor" in rank_config:
            self.penalty_factor = rank_config['penalty_factor']
        if "bert" in rank_config:
            self.bert = rank_config['bert']
        if "similarity_threshold" in rank_config:
            self.similarity_threshold = rank_config['similarity_threshold']
        if "similar_documents" in rank_config:
            self.similar_documents = rank_config['similar_documents']
        if "extract_num" in rank_config:
            self.extract_num = rank_config['extract_num']

    def rank_phrases(self, doc, threshold=None):
        if self.rank_mode == 'multi_turn_selection':
            doc_title, doc_abstract, doc_unsorted_phrases = doc['title'], doc[
                'abstract'], doc['unsorted key phrases']
            # total_document = doc_title + ' ' + doc_abstract
            total_document = 'Title: ' + doc_title + '\n' + 'Abstract: ' + doc_abstract
            all_documents = []
            all_documents.extend(self.similar_documents)
            all_documents.append(total_document)
            doc_unsorted_phrases = remove_repeated(doc_unsorted_phrases)
            candidate_phrases = []
            candidate_phrases.extend(doc_unsorted_phrases)
            phrases_scores_dict_p = collections.defaultdict(int)
            phrases_scores_dict_n = collections.defaultdict(int)
            # min_chosen_num = math.ceil(len(candidate_phrases) / 3)
            # mid_chosen_num = math.ceil(len(candidate_phrases) / 2)
            # max_chosen_num = math.ceil(len(candidate_phrases) * 0.7)
            min_chosen_num = 5 if 5 < len(doc_unsorted_phrases) else math.ceil(len(doc_unsorted_phrases) / 2)
            # min_chosen_num = 5
            mid_chosen_num = 5
            # max_chosen_num = 5
            max_chosen_num = self.extract_num if len(doc_unsorted_phrases) > self.extract_num else len(
                doc_unsorted_phrases)

            # chosen_nums_p = [min_chosen_num] * self.rank_rounds
            chosen_nums_p = [min_chosen_num, min_chosen_num, max_chosen_num]
            chosen_nums_n = [len(doc_unsorted_phrases) - min_chosen_num, len(doc_unsorted_phrases) - min_chosen_num,
                             len(doc_unsorted_phrases) - max_chosen_num]
            # chosen_nums_n = chosen_nums_p
            rank_rounds = self.rank_rounds * 2
            rounds_phrases_list_positive = []
            rounds_phrases_list_negative = []

            for index in tqdm.tqdm(range(rank_rounds), colour='red', desc='排序', maxinterval=100):
                if index < math.ceil(rank_rounds / 2):
                    random.shuffle(candidate_phrases)
                    nums = str(random.choice(chosen_nums_p))
                    # nums = str(chosen_nums_p[index])
                    # selected_document = random.choice(all_documents)
                    # candidate_phrases = random.sample(candidate_phrases,math.floor(len(candidate_phrases)*0.7))
                    candidate_phrases_group_str = '(' + ','.join(["'" + item + "'" for item in candidate_phrases]) + ')'
                    prefix_prompt = self.rank_prompt_positive.replace('<CANDIDATE>',
                                                                      candidate_phrases_group_str).replace(
                        '<DOCUMENT>', total_document).replace('<CHOSEN_NUM>', nums)
                    response = llm_chat(self.model, prompt=prefix_prompt, url=self.LLM_url)

                    try:
                        response_txt = response.replace('\n', '').strip('\t').strip(' ').replace("'", '')
                        response_phrases = json_extract(response_txt)

                        overlap_phrases = set(response_phrases) & set(candidate_phrases)
                        print('选择的短语positive：', overlap_phrases)
                        rounds_phrases_list_positive.append(list(overlap_phrases))

                        for phrase in overlap_phrases:
                            phrases_scores_dict_p[phrase] += 1 * self.reward_factor
                    except:
                        pass
                else:
                    random.shuffle(candidate_phrases)
                    nums = str(random.choice(chosen_nums_n))
                    # nums = str(chosen_nums_n[index - self.rank_rounds])
                    # selected_document = random.choice(all_documents)
                    # candidate_phrases = random.sample(candidate_phrases,math.floor(len(candidate_phrases)*0.7))
                    candidate_phrases_group_str = '(' + ','.join(["'" + item + "'" for item in candidate_phrases]) + ')'
                    prefix_prompt = self.rank_prompt_negative.replace('<CANDIDATE>',
                                                                      candidate_phrases_group_str).replace(
                        '<DOCUMENT>', total_document).replace('<CHOSEN_NUM>', nums)
                    response = llm_chat(self.model, prompt=prefix_prompt, url=self.LLM_url)
                    try:
                        response_txt = response.replace('\n', '').strip('\t').strip(' ')
                        response_phrases = json_extract(response_txt)

                        overlap_phrases = set(response_phrases) & set(candidate_phrases)
                        print('选择的短语negative：', overlap_phrases)
                        rounds_phrases_list_negative.append(list(overlap_phrases))

                        for phrase in overlap_phrases:
                            phrases_scores_dict_n[phrase] += -1 * self.penalty_factor
                    except:
                        pass
                    pass

            phrases_scores_dict_p = dict(phrases_scores_dict_p)
            phrases_scores_dict_n = dict(phrases_scores_dict_n)
            if 'none' in phrases_scores_dict_p:
                del phrases_scores_dict_p['none']
            if 'None' in phrases_scores_dict_p:
                del phrases_scores_dict_p['None']
            if 'none' in phrases_scores_dict_n:
                del phrases_scores_dict_n['none']
            if 'None' in phrases_scores_dict_n:
                del phrases_scores_dict_n['None']

            not_selected_phrases = set(candidate_phrases) - set(list(phrases_scores_dict_p.keys())) - set(
                list(phrases_scores_dict_n.keys()))
            not_selected_phrases_scores = {key: 0 for key in not_selected_phrases}
            phrases_scores_dict_p.update(not_selected_phrases_scores)
            phrases_scores_dict_n.update(not_selected_phrases_scores)
            return phrases_scores_dict_p, phrases_scores_dict_n
