import json
import logging
import os
import re
import yaml

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(g) for g in config['gpu_no']])

if config['debug']:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'data_reader.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def read_metaqa(dataset_name):
    """ Read MetaQA dataset, parse each line of data to (question, answer) pair

    Arg:
        dataset_name(str): one options in the following list
                           [metaqa, metaqa_1hop, metaqa_2hop, metaqa_3hop]

    Return:
        qa_pairs(list of tuple): each element in the list is a tuple including
                                 (question, answer)
    """
    assert dataset_name in ['metaqa', 'metaqa_1hop_train',
                            'metaqa_2hop_train', 'metaqa_3hop_train',
                            'metaqa_1hop_dev', 'metaqa_2hop_dev',
                            'metaqa_3hop_dev']

    file_path = config['_'.join([dataset_name, 'path'])]
    qa_pairs = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            question, answers = line.strip().split('\t')

            question = question.capitalize()
            match = re.search('\[(.*)\]', question)
            entity = match.group()
            new_entity = entity[1:-1]
            question = question.replace(entity, new_entity)

            qa_pairs.append((question, answers))

    return qa_pairs

def read_hotpotqa(dataset_name):
    file_path = config['_'.join([dataset_name, 'path'])]
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    qa_pairs = []
    for data in all_data:
        qa_pairs.append((data['_id'], data['question'], data['answer']))

    return qa_pairs

def read_2wikimultihopqa(dataset_name):
    file_path = config['_'.join([dataset_name, 'path'])]
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    qa_pairs = []
    for data in all_data:
        qa_pairs.append((data['_id'], data['question'], data['answer']))

    return qa_pairs

def read_data(dataset_name):
    """ Return formatted data given dataset name

    Arg:
        dataset_name(str): the name of dataset

    Return:
        data: formatted dataset
    """
    logger.info('Reading ' + dataset_name)
    if 'metaqa' == dataset_name[:6]:
        data = read_metaqa(dataset_name)
    elif 'hotpotqa' == dataset_name[:8]:
        data = read_hotpotqa(dataset_name)
    elif '2wikimultihopqa' == dataset_name[:15]:
        data = read_2wikimultihopqa(dataset_name)
    logger.info('Total # of data: %d'%(len(data)))
    logger.info('Finish reading ' + dataset_name)
    return data

if __name__ == '__main__':
    qa_pairs = read_data(config['dataset_name'])
    print(len(qa_pairs))
    print(qa_pairs[0])
    question, _ = qa_pairs[0]
    from src.utils import clean_string
    print(clean_string(question))

