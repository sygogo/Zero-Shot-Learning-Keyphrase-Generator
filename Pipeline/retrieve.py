import json
from Sentence_bert.text_embedding import text_embedding_tool
from utility import phrase_stmmer, stem_extraction_
from Database.milvus_tools import *


class Retriever:
    def __init__(self, retrieve_config: dict):
        """初始化参数为待检索的Milvus向量数据库中的集合名称以及文档embedding的函数"""
        if "database_ip" in retrieve_config:
            self.database_ip = retrieve_config['database_ip']
        if "database_port" in retrieve_config:
            self.database_port = retrieve_config['database_port']
        if "collection_name" in retrieve_config:
            self.collection_name = retrieve_config['collection_name']
        if "text_embedding" in retrieve_config:
            self.text_embedding = retrieve_config['text_embedding']
        if "retrieve_limit" in retrieve_config:
            self.retrieve_limit = retrieve_config['retrieve_limit']
        if "retrieve_num" in retrieve_config:
            self.retrieve_num = retrieve_config['retrieve_num']
        if "retrieve_threshold" in retrieve_config:
            self.retrieve_threshold = retrieve_config['retrieve_threshold']
        if "extractor" in retrieve_config:
            self.extractor = retrieve_config['extractor']

        self.collection = load_collection(self.database_ip, self.database_port, self.collection_name)

    def release_database(self):
        self.collection.release()

    def document_index_retrieve(self, title: str, abstract: str):
        """先使用待抽取的文档，在向量数据库中检索出若干相似的文档，然后使用大模型来提问"""
        assert self.retrieve_limit >= 10
        to_be_retrieved_doc = title + ' ' + abstract
        to_be_extracted_document_vector = self.text_embedding(to_be_retrieved_doc)
        searched_dicts, searched_dicts_threshold, scores_result = search_data(self.collection,
                                                                              [to_be_extracted_document_vector],
                                                                              result_num=self.retrieve_limit,
                                                                              threshold=self.retrieve_threshold)
        if not searched_dicts_threshold:
            return [], []
        similar_documents = []
        for i in range(len(searched_dicts_threshold)):
            dict_data = json.loads(searched_dicts_threshold[i])  # 使用阈值来过滤
            score = scores_result[i]
            title_ = dict_data['title'].lower().strip('.').strip('?').strip()
            # 去掉相同的文档
            if score >= 0.99 or title.lower().strip('.').strip('?').strip() == title_:
                continue
            if i < 10:
                similar_documents.append({'title': dict_data['title'], 'abstract': dict_data['abstract'], 'sorted key phrases': dict_data['keyphrases']})
        similar_docs = []
        if similar_documents:
            similar_phrases = []
            for similar_document in similar_documents:
                doc = "Title: " + similar_document['title'] + "\n" + "Abstract: " + similar_document['abstract']
                similar_docs.append(doc)
                similar_document_keyphrases = self.extractor.extract_phrases(doc)
                similar_phrases.extend(similar_document_keyphrases)
            similar_phrases_stem = [stem_extraction_(item, phrase_stmmer) for item in similar_phrases]
            similar_phrases_stem_set = set(similar_phrases_stem)
            similar_phrases_stem_freq = {key: similar_phrases_stem.count(key) for key in similar_phrases_stem_set}
            similar_phrases_stem_freq = {key: value for key, value in similar_phrases_stem_freq.items() if value > 1}
            similar_phrases_stem_freq = dict(sorted(similar_phrases_stem_freq.items(), key=lambda i: i[1], reverse=True))
            retrieved_phrases = [similar_phrases[similar_phrases_stem.index(item)] for item in similar_phrases_stem_freq.keys()][:10]
            return retrieved_phrases, similar_docs
        else:
            return [], []
