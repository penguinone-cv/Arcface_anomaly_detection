import csv
from tqdm import tqdm
"""
csvファイルから各種パラメータを一括読み込みする関数
Argument
csv_path: csvファイルのパス
"""
def read_parameters(csv_path, index):
    with open(csv_path, encoding='utf-8-sig') as f: #utf-8-sigでエンコードしないと1列目のキーがおかしくなる
        reader = csv.DictReader(f)
        l = [row for row in tqdm(reader)]
        parameters_dict = l[index]

    return parameters_dict
