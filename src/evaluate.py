import json
import pickle
import yaml
# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)

import wikimultihop_evaluate
from utils import clean_string, get_file_path

dataset_file_path = config['_'.join([config['dataset_name'], 'path'])]
id_aliases_path = config['2wikimultihopqa_id_aliases_path']
new_data = {'answer': {}, 'sp': {}, 'evidence': {}}
path = 'test.json'

#with open(dataset_file_path, 'r') as f:
#    all_data = json.load(f)
#for data in all_data:
#    new_data['answer'][data['_id']] = data['answer']

#file_path = get_file_path(config['context_qa_tuple_text_pickle'],
file_path = get_file_path(config['tuple_ids_pickle'],
                          category='fetching_results')
with open(file_path, 'rb') as f:
    all_data = pickle.load(f)
for data in all_data:
    new_data['answer'][data[0]] = data[-1]

with open(path, 'w') as f:
    json.dump(new_data, f)

wikimultihop_evaluate.eval(path, dataset_file_path, id_aliases_path)
