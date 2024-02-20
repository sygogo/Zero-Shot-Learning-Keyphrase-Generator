import collections
import os
import pickle
import tqdm
from Pipeline.retrieve import Retriever
from Pipeline.extract import LLM_Extractor
from Pipeline.rank import LLM_Ranker
from Pipeline.extend import LLM_Extender
from Pipeline.polish import LLM_Polisher
from utility import *
import argparse
from LLM_tools.LLM_chat import Sentence_Embedding
from transformers import set_seed


def main():
    root_path = config['datasets_root']
    document_path = os.path.join(root_path, f"{args.dataset_name}/word_{args.dataset_name}_testing_context.txt")
    keywords_path = os.path.join(root_path, f"{args.dataset_name}/word_{args.dataset_name}_testing_allkeywords.txt")
    with open(document_path, 'r') as document_files:
        documents = document_files.readlines()
    with open(keywords_path, 'r') as keyword_files:
        keywords = keyword_files.readlines()

    document_keyphrases_dict = collections.defaultdict()

    # 配置短语抽取器
    extract_config = {"extract_prompt": args.extract_prompt, "LLM_url": llm_url, "seed": args.seed,
                      "model": args.model, "extract_num": args.extract_num}
    extractor = LLM_Extractor(extract_config)
    # 配置短语扩展器
    extend_config = {"extend_prompt": args.extend_prompt, "extend_mode": args.extend_mode, "LLM_url": llm_url, "model": args.model,
                     "extend_example_keywords": args.extend_example_keywords,
                     "extend_example_document": args.extend_example_document}
    extender = LLM_Extender(extend_config=extend_config)
    # 配置短语检索器
    retrieve_config = {"database_ip": config["retrieve_database_ip"], "database_port": config["retrieve_database_port"],
                       "collection_name": config["retrieve_collection_name"], "text_embedding": bert.text_embedding,
                       "retrieve_limit": args.retrieve_limit,
                       "retrieve_num": args.retrieve_num, "retrieve_threshold": args.retrieve_similarity,
                       "extractor": extractor}
    retriever = Retriever(retrieve_config)
    # 配置文本优化器
    polish_config = {"polish_prompt": args.polish_prompt, "polish_rounds": args.polish_rounds,
                     "extract_prompt": args.extract_prompt, "LLM_url": llm_url, "extractor": extractor,
                     "model": args.model}
    polisher = LLM_Polisher(polish_config=polish_config)

    # 配置短语排序器
    rank_config = {"rank_rounds": args.rank_rounds,
                   "rank_mode": args.rank_mode, "LLM_url": llm_url, "seed": args.seed,
                   "text_embedding": bert.text_embedding, "model": args.model, "extract_num": args.extract_num,
                   "reward_factor": args.reward_factor, "penalty_factor": args.penalty_factor, 'bert': bert}

    for index, document in tqdm.tqdm(enumerate(documents), desc='读取文档和短语，实施检索，抽取，扩展和排序',
                                     colour='red'):
        # 解析文档，预处理，提取标签短语
        document = document.replace('<digit>', '').replace('( <digit> )', '').replace('( w <digit> )', '').replace(
            '( w > <digit> )', '')
        document_title = document.split('<eos>')[0].strip(' ').strip('\n')
        document_abstract = document.split('<eos>')[1].strip(' ').strip('\n')
        total_document = 'Title: ' + document_title + '\n' + 'Abstract: ' + document_abstract
        label_phrases = keywords[index].split(';')
        label_phrases = [item.strip('\n').replace('\t', '') for item in label_phrases if
                         not item.__contains__('<digit>')]

        # 将标签短语分成present和absent并保存
        label_phrases_p, label_phrases_a = get_present_absent_phrases(total_document, label_phrases)
        document_keyphrases_dict[str(index)] = collections.defaultdict()
        document_keyphrases_dict[str(index)]['document'] = total_document
        document_keyphrases_dict[str(index)]['label_present'] = label_phrases_p
        document_keyphrases_dict[str(index)]['label_absent'] = label_phrases_a

        print(f"\n------------------ 文档{index} 第一步，检索相似文档-------------------")
        retrieved_phrases, similar_documents = retriever.document_index_retrieve(document_title, document_abstract)
        if retrieved_phrases:
            document_keyphrases_dict[str(index)]['retrieve'] = retrieved_phrases
            print("retrieved_phrases: ", retrieved_phrases)
        else:
            print(f"------------------------- 文档{index} 检索失败，使用polish抽取短语-------------------")
            polished_phrases, polished_documents = polisher.polish_document(total_document)
            document_keyphrases_dict[str(index)]['retrieve'] = polished_phrases
            print('retrieved_phrases: ', polished_phrases)
            retrieved_phrases = polished_phrases
            similar_documents = polished_documents

        print(f"------------------ 文档{index} 第二步，抽取关键短语-------------------")
        extracted_phrases = extractor.extract_phrases(total_document)
        print("extracted_phrases: ", extracted_phrases)
        document_keyphrases_dict[str(index)]['extract'] = extracted_phrases

        print(f"---------------文档{index} 第三步, 拓展关键短语-------------------")
        extended_phrases = extender.extend_phrases(target_phrases=extracted_phrases, target_document=total_document)
        document_keyphrases_dict[str(index)]['extend'] = extended_phrases

        # 汇总
        retrieve_extract_extend_phrases = []
        retrieve_extract_extend_phrases.extend(retrieved_phrases)
        retrieve_extract_extend_phrases.extend(extracted_phrases)
        retrieve_extract_extend_phrases.extend(extended_phrases)
        retrieve_extract_extend_phrases = remove_repeated(retrieve_extract_extend_phrases)  # 词干去重
        document_keyphrases_dict[str(index)]['retrieve_extract_extend'] = retrieve_extract_extend_phrases

        # 分成present和absent
        present_phrases, absent_phrases = get_present_absent_phrases(total_document, retrieve_extract_extend_phrases)

        if 'present_phrases_un_rank' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['present_phrases_un_rank'] = present_phrases
        if 'absent_phrases_un_rank' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['absent_phrases_un_rank'] = absent_phrases

        print(f"---------------文档{index} present phrases 排序-------------------")
        rank_config.update({"rank_prompt_positive": args.rank_prompt_present_positive,
                            "rank_prompt_negative": args.rank_prompt_present_negative,
                            "similar_documents": similar_documents})
        ranker = LLM_Ranker(rank_config)
        ranked_present_phrases_scores_p, ranked_present_phrases_scores_n = ranker.rank_phrases(
            {'title': document_title, 'abstract': document_abstract, 'unsorted key phrases': present_phrases})

        if len(absent_phrases) >= 1:
            print(f"---------------文档{index} absent phrases 排序-------------------")
            rank_config.update({"rank_prompt_positive": args.rank_prompt_absent_positive,
                                "rank_prompt_negative": args.rank_prompt_absent_negative,
                                "similar_documents": similar_documents})
            ranker = LLM_Ranker(rank_config)
            ranked_absent_phrases_scores_p, ranked_absent_phrases_scores_n = ranker.rank_phrases(
                {'title': document_title, 'abstract': document_abstract, 'unsorted key phrases': absent_phrases})
        else:
            ranked_absent_phrases_scores_p = dict()
            ranked_absent_phrases_scores_n = dict()

        # 整合
        present_ranked_scores = dict()
        absent_ranked_scores = dict()
        present_ranked_scores.update(ranked_present_phrases_scores_p)
        present_ranked_scores_relevant = dict(sorted(present_ranked_scores.items(), key=lambda i: i[1], reverse=True))
        if ranked_present_phrases_scores_n:
            for key, value in ranked_present_phrases_scores_n.items():
                if key not in present_ranked_scores:
                    present_ranked_scores.update({key: value})
                else:
                    present_ranked_scores[key] += value
        present_ranked_scores = dict(sorted(present_ranked_scores.items(), key=lambda i: i[1], reverse=True))
        absent_ranked_scores.update(ranked_absent_phrases_scores_p)
        absent_ranked_scores_relevant = dict(sorted(absent_ranked_scores.items(), key=lambda i: i[1], reverse=True))
        if ranked_absent_phrases_scores_n:
            for key_, value_ in ranked_absent_phrases_scores_n.items():
                if key_ not in absent_ranked_scores:
                    absent_ranked_scores.update({key_: value_})
                else:
                    absent_ranked_scores[key_] += value_
        absent_ranked_scores = dict(sorted(absent_ranked_scores.items(), key=lambda i: i[1], reverse=True))

        # # 过滤
        # present_ranked_scores = {key: value for key, value in present_ranked_scores.items()}
        # absent_ranked_scores = {key: value for key, value in absent_ranked_scores.items()}
        # present_ranked_scores_relevant = {key: value for key, value in present_ranked_scores_relevant.items()}
        # absent_ranked_scores_relevant = {key: value for key, value in absent_ranked_scores_relevant.items()}

        # 归一化并保存
        present_ranked_scores = get_softmax_result(present_ranked_scores)
        absent_ranked_scores = get_softmax_result(absent_ranked_scores)
        present_ranked_scores_relevant = get_softmax_result(present_ranked_scores_relevant)
        absent_ranked_scores_relevant = get_softmax_result(absent_ranked_scores_relevant)
        if 'present_ranked_scores' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['present_ranked_scores'] = present_ranked_scores
            print('present_ranked_scores: ', present_ranked_scores)
        if 'absent_ranked_scores' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['absent_ranked_scores'] = absent_ranked_scores
            print('absent_ranked_scores: ', absent_ranked_scores)
        if 'present_ranked_scores_relevant' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['present_ranked_scores_relevant'] = present_ranked_scores_relevant
        if 'absent_ranked_scores_relevant' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['absent_ranked_scores_relevant'] = absent_ranked_scores_relevant

        # 将present和absent的数据融合起来并排序
        final_result = dict()
        final_result.update(present_ranked_scores)
        final_result.update(absent_ranked_scores)
        final_result = dict(sorted(final_result.items(), key=lambda i: i[1], reverse=True))
        if 'final_result' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['final_result'] = list(final_result.keys())

        # 仅保留relevant prompt的结果
        final_result_ = dict()
        final_result_.update(present_ranked_scores_relevant)
        final_result_.update(absent_ranked_scores_relevant)
        final_result_ = dict(sorted(final_result_.items(), key=lambda i: i[1], reverse=True))
        if 'final_result_relevant' not in document_keyphrases_dict:
            document_keyphrases_dict[str(index)]['final_result_relevant'] = list(final_result_.keys())

        parameters = [args.model, str(args.seed), str(args.retrieve_limit), str(args.retrieve_similarity),
                      str(args.retrieve_num), str(args.rank_rounds), str(args.extract_num)]
        saved_file_name = "_".join(parameters) + "_" + str(args.dataset_name) + "_" + str(
            args.model) + "_results" + ".pkl"

        save_path = os.path.join(config['results_root'], args.dataset_name, saved_file_name)

        os.makedirs(os.path.dirname(os.path.join(config['results_root'], args.dataset_name)), exist_ok=True)

        if index == 0:
            document_keyphrases_dict['prompts'] = {"present_positive": args.rank_prompt_present_positive,
                                                   "present_negative": args.rank_prompt_present_negative,
                                                   "absent_positive": args.rank_prompt_absent_positive,
                                                   "absent_negative": args.rank_prompt_absent_negative}
            document_keyphrases_dict['desc'] = {"desc": "present只保留抽取的，absent保留所有的"}
        document_keyphrases_dict = {key: dict(value) for key, value in document_keyphrases_dict.items()}
        with open(save_path, 'wb') as f:
            pickle.dump(document_keyphrases_dict, f)


if __name__ == '__main__':
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # extract_nums = {"semeval": 14, "krapivin": 6, "nus": 10, "inspec": 9, "kp20k": 6}
    # bert = text_embedding_tool(config["bert_path"])
    ber_url = "http://" + config['sentence_transformers_ip'] + ":" + str(config['sentence_transformers_port'])
    bert = Sentence_Embedding(ber_url)
    parser = argparse.ArgumentParser(description='数据集的运行代码')
    parser.add_argument('--dataset_name', type=str, default='semeval')
    parser.add_argument('--model', type=str, default='openchat')
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--retrieve_limit', type=int, default=50)
    parser.add_argument('--retrieve_similarity', type=float, default=0.8)
    parser.add_argument('--retrieve_num', type=int, default=10)
    parser.add_argument('--polish_prompt', type=str, default=config['polish_prompt_template'])
    parser.add_argument('--polish_rounds', type=int, default=5)
    parser.add_argument('--extract_prompt', type=str, default=config['extract_prompt_template'])
    parser.add_argument('--extract_num', type=int, default=14)
    parser.add_argument('--extend_prompt', type=str, default=config['extend_prompt_template'])
    parser.add_argument('--extend_mode', type=str, default='context_extend')
    parser.add_argument('--rank_mode', type=str, default='multi_turn_selection')
    parser.add_argument('--reward_factor', type=float, default=1.0)
    parser.add_argument('--penalty_factor', type=float, default=0.5)
    parser.add_argument('--rank_rounds', type=int, default=5)
    parser.add_argument('--rank_prompt_present_positive', type=str,
                        default=config['rank_prompt_template_positive'])
    parser.add_argument('--rank_prompt_present_negative', type=str,
                        default=config['rank_prompt_template_negative'])
    parser.add_argument('--rank_prompt_absent_positive', type=str,
                        default=config['rank_prompt_template_positive'])
    parser.add_argument('--rank_prompt_absent_negative', type=str,
                        default=config['rank_prompt_template_negative'])

    args = parser.parse_args()
    set_seed(args.seed)
    llm_url = ''
    if args.model == 'openchat':
        llm_url = 'http://' + config['openchat_ip'] + ":" + str(config['openchat_port'])
    if args.model == 'vicuna':
        llm_url = 'http://' + config['vicuna_ip'] + ":" + str(config['vicuna_port'])
    if args.model == 'mistral':
        llm_url = 'http://' + config['mistral_ip'] + ":" + str(config['mistral_port'])
    main()
