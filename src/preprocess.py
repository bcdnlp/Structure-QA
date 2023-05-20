import logging
import os
import pickle
import yaml

with open('src/config.yml') as f:
    config = yaml.safe_load(f)

from tqdm import tqdm

from torch.utils.data import Dataset
from transformers import T5Tokenizer

from src.utils import get_file_path, clean_files
from src.dataset_reader import read_data
from src.context_reader import context_reader
from src.context_fetching import context_fetching
from src.knowledge_graph_generation import graph_generation
from src.demonstration import demo_generation

if config['debug']:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'data_processing.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class StructureDataset(Dataset):
    def __init__(self, dataset_name, generate_structure=False):
        # load/read wikidata, dataset, and bm25 pre-fetched data in text format
        logger.info('Initializing dataset')

        # read dataset
        qa_pairs = read_data(dataset_name)

        # read and fetch context
        context_reader()
        context_qa_tuples_text = context_fetching(qa_pairs)
        logging.info('total # of data: ' + str(len(context_qa_tuples_text)))
#        input()

        # construct knowledge graphs
        tuples_text = graph_generation(context_qa_tuples_text)
        logging.info(tuples_text[0])
#        input()

        # convert text to ids and attention mask
        file_path = get_file_path(config['tuple_ids_pickle'],
                                  category='fetching_results')
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                self.context_qa_tuples = pickle.load(f)

            logger.info('Load context question answer tuple ids')

            return

        if 't5' in config['infer_model']:
            tokenizer = T5Tokenizer.from_pretrained(config['infer_model'],
                                                    model_max_length=config['model_max_length'])
        self.context_qa_tuples = []
        for context_qa_tuple in tqdm(tuples_text,
                                     desc='Converting text to ids'):
            if 'none' == config['graph_type']:
                idxes, context_text, question_text, answer_text = context_qa_tuple
            else:
                idxes, context_text, graph_text, question_text, answer_text = context_qa_tuple

            # add contexts
            context = '\n'.join(['Documents:', context_text])
            input_text = context

            # add graph
            if 'empty' == config['graph_type']:
                graph = 'Graph:'
                input_text = '\n\n'.join([input_text, graph])
            elif 'none' != config['graph_type']:
                graph = '\n'.join(['Graph:', graph_text])
                input_text = '\n\n'.join([input_text, graph])

            # add question prompts based on different models
            if 'openai' == config['infer_model']:
                question = '\n'.join(['Question:', question_text])
                input_text = '\n\n'.join([input_text, question])
            elif 't5' in config['infer_model']:
                if config['cot']:
                    question_prompt = 'Answer the following question by reasoning step-by-step.'
                else:
                    question_prompt = 'Answer the following question.'
                question = ' '.join([question_prompt, question_text])
                input_text = '\n\n'.join([input_text, question])

            # add answer prompt
            answer_prompt = 'Answer:\n'
#            answer_prompt = 'output short answer or what other information is required to answer the question:'
            input_text = '\n\n'.join([input_text, answer_prompt])

            # add demonstrations
            if 'few-shot' == config['shot']:
                demos = demo_generation(input_text)
                input_text = ''.join([demos, input_text])

            # convert to ids if use t5 models
            if 't5' in config['infer_model']:
                inputs = tokenizer(input_text)
                input_ids = inputs.input_ids
                attn_mask = inputs.attention_mask
                input_tuple = (idxes, input_ids, attn_mask, question_text, answer_text)
            elif 'openai' == config['infer_model']:
                input_tuple = (idxes, input_text, question_text, answer_text )
            self.context_qa_tuples.append(input_tuple)

        with open(file_path, 'wb') as f:
            pickle.dump(self.context_qa_tuples, f)

    def __len__(self):
        return len(self.context_qa_tuples)

    def __getitem__(self, idx):
        return self.context_qa_tuples[idx]

if __name__ == '__main__':
    clean_files()
    dataset = StructureDataset(config['dataset_name'])

