#!/bin/bash

# 启动 Python 脚本1
#python run.py --dataset_name nus

# 启动 Python 脚本2
#python run.py --dataset_name krapivin

# 启动 Python 脚本3
#python run.py --dataset_name semeval
python run.py --dataset_name semeval --extract_num 14 --model openchat --rank_rounds 1
python run.py --dataset_name krapivin --extract_num 6 --model openchat --rank_rounds 1
python run.py --dataset_name inspec --extract_num 9 --model openchat --rank_rounds 1
python run.py --dataset_name nus --extract_num 10 --model openchat --rank_rounds 1

