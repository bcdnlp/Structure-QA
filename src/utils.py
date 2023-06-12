import os
import re
import string
import logging
import yaml

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'utils_.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

nlp = spacy.load('en_core_web_sm')

def clean_string(text, is_question=False):
    final_string = ""

    # Make lower
    text = text.lower()

    # Remove url links
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'&lt.*&gt', '', text)

    # Remove puncuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    if not is_question:
        # Remove stop words
        text = text.split()
        useless_words = STOP_WORDS
        text = [word for word in text if not word in useless_words]
        text = ' '.join(text)

    text = re.sub(r'[ \t]+', ' ', text)
    return text

def get_file_path(file_name, category, folder_flag=False):
    dataset_name = config['dataset_name']
    # get file name as file name prefix
    path = config['_'.join([dataset_name, 'path'])]
    dataset_file_name = os.path.splitext(os.path.basename(path))[0]
    # extract dataset name for saving folder
    if 'metaqa' == dataset_name[:6]:
        dataset_name = 'metaqa'
    elif 'hotpotqa' == dataset_name[:8]:
        dataset_name = 'hotpotqa'
    elif '2wikimultihopqa' == dataset_name[:15]:
        dataset_name = '2wikimultihopqa'

    if 'context_reader' == category:
        if 'faiss' == config['retriever'][:5]:
            folder = os.path.join(config['intermediate_folder'],
                                  config['context'],
                                  'faiss')
        else:
            folder = os.path.join(config['intermediate_folder'],
                                  config['context'],
                                  config['bm25_tokenizer'])
        if not os.path.exists(folder):
            os.makedirs(folder)
 
        if folder_flag:
            path = folder
        else:
            path = os.path.join(folder, file_name)
    elif 'context_index' == category:
        folder = os.path.join(config['intermediate_folder'],
                              config['retriever'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        if 'bm25' == config['retriever']:
            file_name = '_'.join([config['bm25_tokenizer'], file_name])
        if folder_flag:
            path = folder
        else:
            path = os.path.join(folder, file_name)
    elif 'fetching_results' == category:
        folder = os.path.join(config['intermediate_folder'],
                              dataset_name,
                              config['retriever'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        # baseline
        file_name_prefix = dataset_file_name
        if 'none' != config['graph_type']:
            # kg1
            file_name_prefix = '_'.join([file_name_prefix,
                                         config['graph_type'],
                                         config['ner_model']])
            if config['relation']:
                # kg2
                rel_model = config['rel_model']
                if 'google' == rel_model[:6]:
                    rel_model = rel_model[7:]
                file_name_prefix = '_'.join([file_name_prefix,
                                             rel_model])

        shot = config['shot']
        file_name_prefix = '_'.join([file_name_prefix, shot])

#        if config['cot']:
#            file_name_prefix = '_'.join([file_name_prefix, 'cot'])
#
        infer_model = config['infer_model']
        if 't5' in infer_model:
            infer_model = 't5'
        file_name_prefix = '_'.join([file_name_prefix, infer_model])
        file_name = '_'.join([file_name_prefix, file_name])
        if folder_flag:
            path = folder
        else:
            path = os.path.join(folder, file_name)
    elif 'results' == category:
        folder = os.path.join(config['results_dir'],
                              dataset_name,
                              config['infer_model'])
        if not os.path.exists(folder):
            os.makedirs(folder)

        # baseline
        file_name_prefix = '_'.join([dataset_file_name,
                                     config['mode'],
                                     config['retriever']])
        if config['retriever'] not in ['golden', 'manual']:
            file_name_prefix = '_'.join([file_name_prefix, str(config['top_n'])])
        if 'none' != config['graph_type']:
            # kg1
            file_name_prefix = '_'.join([file_name_prefix,
                                         config['graph_type'],
                                         config['ner_model']])
            if 'none' != config['relation']:
                # kg2
                rel_model = config['rel_model']
                if 'google' == rel_model[:6]:
                    rel_model = rel_model[7:]
                file_name_prefix = '_'.join([file_name_prefix,
                                             rel_model])

        shot = config['shot']
        file_name_prefix = '_'.join([file_name_prefix, shot])

        if config['cot']:
            file_name_prefix = '_'.join([file_name_prefix, 'cot'])

        infer_model = config['infer_model']
        if 'google' == infer_model[:6]:
            infer_model = infer_model[7:]
        file_name_prefix = '_'.join([file_name_prefix, infer_model])
        file_name = '_'.join([file_name_prefix, file_name])
        if folder_flag:
            path = folder
        else:
            path = os.path.join(folder, file_name)
    else:
        raise Error
    return path

def clean_files():
    if 'none' == config['override']:
        pass
    elif 'all' == config['override']:
        logging.info('Removing: %s'%(config['intermediate_folder']))
        input()
        if os.path.exists(config['intermediate_folder']):
            os.rmdir(config['intermediate_folder'])
    elif config['override'] in ['context_reader', 'context_index']:
        folder_path = get_file_path('', category=config['override'], folder_flag=True)
        logging.info('Removing: %s'%(folder_path))
        y = input('Confirm deleting the folder? (y or n)')
        if y in ['y', 'Y'] and os.path.exists(folder_path):
            os.rmdir(folder_path)
    elif config['override'] in ['fetched_results',
                                'fetched_entity',
                                'fetched_graph',
                                'preprocessed_tuple']:
        # remvoe tuple ids
        file_path = get_file_path(config['tuple_ids_pickle'],
                                  category='fetching_results')
        if os.path.exists(file_path):
            os.remove(file_path)

        # remove graphs
        if config['override'] in ['fetched_results', 'fetched_entity', 'fetched_graph']:
            file_path = get_file_path(config['context_graph_qa_tuple_text_pickle'],
                                      category='fetching_results')
            if os.path.exists(file_path):
                os.remove(file_path)

        # remove entities
        if config['override'] in ['fetched_results', 'fetched_entity']:
            file_path = get_file_path(config['context_entity_qa_tuple_text_pickle'],
                                      category='fetching_results')
            if os.path.exists(file_path):
                os.remove(file_path)

        # remove contexts qa results
        if 'fetched_results' == config['override']:
            file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                                      category='fetching_results')
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        raise Error

    logger.info('Clean Finished')

if __name__ == '__main__':
    print(clean_string('ab, c. d', False))
    print(get_file_path('a.dat', category='context_reader'))

