import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from sentence_transformers import SentenceTransformer


class text_embedding_tool:
    def __init__(self, weight_path):
        self.model = SentenceTransformer(weight_path)
        self.model.cuda()

    def get_text_embeddings(self, text_list):
        embeddings_list = self.model.encode(text_list, show_progress_bar=False, device='cuda:0')
        return embeddings_list


if __name__ == '__main__':
    path = '/home/daishengran/deploy/Models/sentence_transformers'
    the_model = text_embedding_tool(path)
    result = the_model.get_text_embeddings(['hello', 'ok'])
    print(result)
