The code for the paper "Thinking Like an Author: A Zero-shot Learning Approach to Keyphrase Generation with Large Language Model". Our paper has been accepted by ECML-PKDD 2024.

# Thinking Like an Author: A Zero-shot Learning Approach to Keyphrase Generation with Large Language Model
## **1. Install Requirements**
`pip install -r requirements.txt`

## **2. Start Sentence_Transformers Server**
download the sentence_transformers model weight from: [sentence_transformers](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)   
then modify the *bert_path*, *sentence_transformers_ip* and *sentence_transformers_port* in *config.json*   

use`python Sentence_bert/sentence_transformers_server.py` to start a sentence_transformers to get text embedding vectors
    
## **3. Build Kp20k Train Database for Retrieve**
download the kp20k train data file from: [kp20k](https://huggingface.co/datasets/taln-ls2n/kp20k/tree/main)   
then set the *kp20k_train_path*, *retrieve_database_ip* and *retrieve_database_port* in *config.json*   

use`python Database/kp20k_collection.py` to full the database with kp20k train datas

## **4. Start Openchat Server**
download openchat model weights from: [openchat](https://github.com/imoneoi/openchat/tree/master)

set the *openchat_path*, *openchat_ip* and *openchat_port* in *config.json* and then use `python LLM_tools/openchat_7b_server.py` to start the server

## **5. Run**
    python run.py --dataset_name semeval --model openchat --retrieve_limit 50 --retrieve_similarity 0.8 --retrieve_num 10 --extract_num 14 --rank_rounds 5   

## **6. Evaluation**
    python evaluation.py --dataset semeval --coefficient 14.15  

## **7. BaseLine Models**
### ** vicuna **
download vicuna model weights from: [vicuna](https://huggingface.co/lmsys/vicuna-7b-v1.5)

set the *vicuna_path*, *vicuna_ip* and *vicuna_port* in *config.json* and then use`python LLM_tools/vicuna_7b_server.py` to start the server   

### ** mistral **
download mistral model weights from: [mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

set the *mistral_path*, *mistral_ip* and *mistral_port* in *config.json* and then use`python LLM_tools/mistral_7b_server.py` to start the server  

### ** chatgpt **
export your *api_key*, *azure_endpoint* and *api_version* to environment variables profile

### run baseline models to extract
`python baseline/baseline_models_extract.py --dataset_name semeval --model mistral --extract_num 14`
### evaluate baseline models extract results
`python evaluation_.py --dataset semeval`
