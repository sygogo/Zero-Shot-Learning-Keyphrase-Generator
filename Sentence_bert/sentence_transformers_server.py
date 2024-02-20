import os
import json
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
app = Flask(__name__)
weight_path = '/data/daishengran/sentence_transformers'
bert = SentenceTransformer(weight_path)
bert.cuda()


@app.route('/', methods=['POST', 'GET'])
def response_func():
    # 获取请求参数
    re_data = request.get_json()
    print(re_data)
    text_list = re_data.get('text_list')
    print(text_list)
    result = bert.encode(text_list, show_progress_bar=False).tolist()
    print(result)
    return jsonify(result)


if __name__ == '__main__':
    with open('../config.json', 'r') as config_file:
        config = json.load(config_file)
    host = config['sentence_transformers_ip']
    port = config['sentence_transformers_port']
    app.run(host=host, port=port)
