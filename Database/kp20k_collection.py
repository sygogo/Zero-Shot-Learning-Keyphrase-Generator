import json

import tqdm

from Sentence_bert.text_embedding import text_embedding_tool
from milvus_tools import *

if __name__ == '__main__':
    """database:default collection:kp20k_train"""
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    milvus_ip = config['retrieve_database_ip']
    milvus_port = config['retrieve_database_port']
    kp20k_train_path = config['kp20k_train_path']
    sentence_bert_path = config['bert_path']
    sentence_bert_model = text_embedding_tool(sentence_bert_path)
    conn = connect_milvus(milvus_ip, milvus_port)
    print("======= 向量数据库连接成功，sentence_bert部署成功 ========\n")
    db.using_database('default')
    drop_collection('kp20k_train')
    collection_kp20k = create_collection('kp20k_train')
    print("============== 创建kp20k_train集合成功 ================\n")

    with open(kp20k_train_path, 'r') as kp20k_json_file:
        for index, line in tqdm.tqdm(enumerate(kp20k_json_file), desc="将kp20k训练集中的json插入数据库中去"):
            document_dict = json.loads(line)
            document_json = json.dumps({"title": document_dict["title"], "abstract": document_dict["abstract"],
                                        "keyphrases": document_dict["keyphrases"]})
            document_text = 'title: ' + document_dict['title'] + 'abstract: ' + document_dict[
                'abstract'] + 'keyphrases: ' + ','.join(document_dict['keyphrases'])
            embedding = sentence_bert_model.get_text_embeddings([document_text])[0].tolist()

            insert_to_collection((index, document_json, embedding), collection_kp20k)

    flush_collection(collection_kp20k)
