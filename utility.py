import json
import re
import numpy as np
import torch
import torch.nn.functional as F
from nltk import PorterStemmer, word_tokenize
from LLM_tools.LLM_chat import Sentence_Embedding
phrase_stmmer = PorterStemmer()


def stem_extraction(phrase: str, stemmer: PorterStemmer):
    words = phrase.split(' ')
    stem_phrase = ' '.join([stemmer.stem(word) for word in words])
    return stem_phrase


def stem_extraction_(text: str, stemmer: PorterStemmer):
    words = word_tokenize(text)
    stems = [stemmer.stem(word) for word in words]
    text_stems = ' '.join(stems)
    return text_stems


def json_extract(text: str):
    key_phrases = []
    full_json_pattern = r'\{[^{}]*\}'
    half_json_pattern = r'\[[^\]]*"([^"]*)"[^\]]*\]'
    zero_json_pattern = r'"([^"]*)"'

    try:
        # full_match = re.match(full_json_pattern, text, re.DOTALL)
        full_match = re.search(full_json_pattern, text, re.DOTALL)
        if full_match:
            try:
                # json_dict = json.loads(full_match.group())
                json_dict = json.loads(full_match.group(0))
                for key, value in json_dict.items():
                    if key == 'explanation':
                        continue
                    else:
                        if type(value) == list:
                            key_phrases.extend(value)
                        if type(value) == str:
                            key_phrases.append(value)

                return list(set(key_phrases))
            except:
                pass
    except:
        pass
    key_phrases = list(set(key_phrases))

    if not key_phrases:
        temp = None
        try:
            temp = text.strip('{').strip('}').strip('[').strip(']').strip(',').strip("'").split(',')
        except:
            pass
        finally:
            if temp:
                key_phrases = temp
    return key_phrases


def json_extract_2(text: str):
    key_phrases_p = []
    key_phrases_n = []
    full_json_pattern = r'\{[^{}]*\}'
    half_json_pattern = r'\[[^\]]*"([^"]*)"[^\]]*\]'
    zero_json_pattern = r'"([^"]*)"'

    try:
        full_match = re.match(full_json_pattern, text, re.DOTALL)
        if full_match:
            try:
                json_dict = json.loads(full_match.group())
                for key, value in json_dict.items():
                    if key.strip() == 'relevant phrases':
                        if type(value) == list:
                            key_phrases_p.extend(value)
                        if type(value) == str:
                            key_phrases_p.append(value)
                    if key.strip() == 'irrelevant phrases':
                        if type(value) == list:
                            key_phrases_n.extend(value)
                        if type(value) == str:
                            key_phrases_n.append(value)
            except:
                pass
    except:
        print(text)
        pass
    return {"relevant phrases": key_phrases_p, "irrelevant phrases": key_phrases_n}


def get_present_absent_phrases(document: str, original_phrases: list[str]):
    document_stem = stem_extraction_(document.replace('-', ' ').replace('  ', ' ').replace('_', ' '), phrase_stmmer)
    original_phrases_stem = [
        stem_extraction_(item.strip().replace('-', ' ').replace('  ', ' ').replace('_', ' '), phrase_stmmer) for item
        in original_phrases]
    original_phrases_p, original_phrases_a = [], []
    if original_phrases_stem:
        for phrase_stem in original_phrases_stem:
            phrase = original_phrases[original_phrases_stem.index(phrase_stem)]
            if document_stem.__contains__(phrase_stem):
                if phrase not in original_phrases_p:
                    original_phrases_p.append(phrase)
            else:
                if phrase not in original_phrases_a:
                    original_phrases_a.append(phrase)
    return original_phrases_p, original_phrases_a


def remove_repeated(phrases_list: list):
    phrases_stem = [stem_extraction_(item, phrase_stmmer) for item in phrases_list]
    phrases_stem_unique = list(set(phrases_stem))
    final_phrases = []
    for phrase_stem in phrases_stem_unique:
        final_phrases.append(phrases_list[phrases_stem.index(phrase_stem)].strip())
    return final_phrases


def get_cosine_similarity(document: str, phrases_list: list, bert: Sentence_Embedding, threshold=0.5):
    document_embedding = bert.text_embedding([document])[0]
    phrases_embeddings = bert.text_embedding(phrases_list)
    document_embedding_t = torch.tensor(document_embedding).view((1, len(document_embedding))).cuda()
    phrases_embeddings_t = torch.tensor(phrases_embeddings).view(
        (len(phrases_embeddings), len(phrases_embeddings[0]))).cuda()
    cosine_similarity = F.cosine_similarity(document_embedding_t, phrases_embeddings_t, dim=1).tolist()

    phrases_scores = {phrases_list[index]: cosine_similarity[index] for index in range(len(phrases_list))}
    phrases_scores = {key: value for key, value in phrases_scores.items() if value > threshold}
    phrases_scores = dict(sorted(phrases_scores.items(), key=lambda i: i[1], reverse=True))
    return phrases_scores


def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return probabilities


def get_softmax_result(ranked_scores: dict):
    p_keys = list(ranked_scores.keys())
    p_values = list(ranked_scores.values())
    if p_values:
        softmax_values = softmax(np.array(p_values))
        ranked_scores = {key: softmax_values[p_keys.index(key)] for key in p_keys}
    return ranked_scores


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    bert = Sentence_Embedding(config['sentence_transformers'])
    doc = 'an empirical evaluation of performance memory trade offs in time warp . <eos> <unk> performance of the ' \
          'time warp mechanism is experimentally evaluated when only a limited amount of memory is available to the ' \
          'parallel computation'
    s = get_cosine_similarity(doc, ['in', 'time warp', 'hello', 'empirical evaluation', 'an empirical evaluation',
                                    'an empirical evaluation of', '<eos>', 'performance'], bert)
    print(s)
