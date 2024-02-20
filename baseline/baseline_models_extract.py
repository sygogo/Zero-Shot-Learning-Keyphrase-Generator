import collections
import os
import pickle

import tqdm
from Pipeline.extract import LLM_Extractor
from utility import *
import argparse
from LLM_tools.LLM_chat import Sentence_Embedding
from transformers import set_seed


def main():
    root_path = '../' + config['datasets_root']
    document_path = os.path.join(root_path, f"{args.dataset_name}/word_{args.dataset_name}_testing_context.txt")
    keywords_path = os.path.join(root_path, f"{args.dataset_name}/word_{args.dataset_name}_testing_allkeywords.txt")

    with open(document_path, 'r') as document_files:
        documents = document_files.readlines()

    with open(keywords_path, 'r') as keyword_files:
        keywords = keyword_files.readlines()

    document_keyphrases_dict = collections.defaultdict()

    # 配置短语抽取器
    extract_config = {"extract_prompt": args.extract_prompt, "seed": args.seed,
                      "model": args.model, "extract_num": args.extract_num, "LLM_url": llm_url}
    extractor = LLM_Extractor(extract_config)

    for index, document in tqdm.tqdm(enumerate(documents), desc='gpt抽取短语', colour='red'):
        # 解析文档
        document = document.replace('<digit>', '').replace('( <digit> )', '').replace('( w <digit> )', '').replace(
            '( w > <digit> )', '')
        document_title = document.split('<eos>')[0].strip(' ').strip('\n')
        document_abstract = document.split('<eos>')[1].strip(' ').strip('\n')
        total_document = 'Title: ' + document_title + '\n' + 'Abstract: ' + document_abstract

        label_phrases = keywords[index].split(';')
        label_phrases = [item.strip('\n').replace('\t', '') for item in label_phrases if
                         not item.__contains__('<digit>')]

        label_phrases_p, label_phrases_a = get_present_absent_phrases(total_document, label_phrases)

        document_keyphrases_dict[str(index)] = collections.defaultdict()
        document_keyphrases_dict[str(index)]['document'] = total_document
        document_keyphrases_dict[str(index)]['label_present'] = label_phrases_p
        document_keyphrases_dict[str(index)]['label_absent'] = label_phrases_a

        extracted_phrases = extractor.extract_phrases(total_document)
        print("extracted_phrases: ", extracted_phrases)
        document_keyphrases_dict[str(index)]['extract'] = extracted_phrases

        parameters = [args.model, str(args.seed), str(args.extract_num)]
        saved_file_name = "_".join(parameters) + "_" + str(args.dataset_name) + "_" + str(
            args.model) + "_extract" + ".pkl"

        save_path = '../' + os.path.join(config['results_root'], args.dataset_name, saved_file_name)

        os.makedirs(os.path.dirname(os.path.join(config['results_root'], args.dataset_name)), exist_ok=True)

        if index == 0:
            document_keyphrases_dict['desc'] = {"desc": "只使用gpt抽取,prompt使用openchat的"}
            document_keyphrases_dict['prompts'] = {"prompt": args.extract_prompt}
        document_keyphrases_dict = {key: dict(value) for key, value in document_keyphrases_dict.items()}
        with open(save_path, 'wb') as f:
            pickle.dump(document_keyphrases_dict, f)


if __name__ == '__main__':
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)

    parser = argparse.ArgumentParser(description='baseline methods')
    parser.add_argument('--dataset_name', type=str, default='semeval')
    parser.add_argument('--model', type=str, default='vicuna')
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--extract_prompt', type=str, default=config['extract_prompt_template'])
    parser.add_argument('--extract_num', type=int, default=14)

    args = parser.parse_args()
    set_seed(args.seed)
    llm_url = ''
    if args.model == 'vicuna':
        llm_url = 'http://' + config['vicuna_ip'] + ":" + str(config['vicuna_port'])
    if args.model == 'mistral':
        llm_url = 'http://' + config['mistral_ip'] + ":" + str(config['mistral_port'])
    if args.model == 'chatgpt':
        llm_url = None
    main()
