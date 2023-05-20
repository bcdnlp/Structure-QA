import json
import logging
import pickle
import os
import yaml

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(g) for g in config['gpu_no']])

from tqdm import tqdm
from multiprocessing import Pool, Manager, Lock

from transformers import T5Tokenizer
from transformers import DPRContextEncoderTokenizer

from utils import clean_string, get_file_path

if config['debug']:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'context_reader.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def init_lock(lck):
    global lock
    lock = lck

def read_enwiki_single_file(file_name, tokenizer, corpus_list, corpus_token_list):
    """ Read a single enwiki articles and tokenize it
        Each document is splitted by lines. Each line is a paragraph. Thus,
        the retriever is built based on paragraph-level retriever.

    Args:
        file_name(str): the file waiting to be parsed and tokenzied
        tokenizer: the tokenizer used to tokenize the file
        corpus_list(list): a list gathering all cleaned documents
        corpus_token_list(list): a list gathering all tokenized documents

    Return:
        count(int): the number of paragraphs that cleaned and tokenized
    """
    count = 0
    with open(file_name, 'r') as f:
        for line in f.readlines():
            content = json.loads(line)
            # remove emtpy documents and short documents
            if 0 == len(content['text'].strip()):
                continue

            # clean corpus
            paragraphs = content['text'].split('\n')
            paragraphs = [paragraph for paragraph in paragraphs
                                    if config['min_doc_length'] < len(paragraph.split())]
            if 0 == len(paragraphs):
                continue
            paragraphs = [clean_string(paragraph) for paragraph in paragraphs]
            count += len(paragraphs) # for statistics

            # tokenize corpus
            if 'bm25' == config['retriever']:
                tokens = tokenizer(paragraphs,
                                   max_length=config['max_doc_length'],
                                   truncation=True).input_id
                tokens = [tokenizer.convert_ids_to_tokens(tokens_id)
                    for tokens_id in tokens_ids]
            elif 'faiss' == config['retriever']:
                tokens = tokenizer(paragraphs,
                                   max_length=config['max_doc_length'],
                                   padding=True,
                                   truncation=True)
                tokens = dict(tokens)
                tokens = [dict(zip(tokens,t)) for t in zip(*tokens.values())]

            # add to corresponding lists
            lock.acquire()
            corpus_list.extend(paragraphs)
            corpus_token_list.extend(tokens)
            lock.release()

    # iteratively store and fetch documents
    if config['save_memory']:
        lock.acquire()
        if config['max_doc_number'] < len(corpus_list):
            logger.info(str(len(corpus_list)))
            save_corpus_list = list(corpus_list)
            path = get_file_path(config['enwiki_corpus_pickle'],
                                 category='context_reader')
            with open(path, 'a+b') as f:
                pickle.dump(save_corpus_list, f)
            save_corpus_token_list = list(corpus_token_list)
            path = get_file_path(config['enwiki_corpus_tokens_pickle'],
                                 category='context_reader')
            with open(path, 'a+b') as f:
                pickle.dump(save_corpus_token_list, f)
            corpus_list[:] = []
            corpus_token_list[:] = []
        lock.release()

    return count

def read_enwiki_parallel():
    """ Read enwiki articles and store tokenized documents on disk

        corpus_token_list(2-D list):
            each element in the outer list is a list of tokens
            from a document. The total file is too large to be
            calculated in each run. It is stored on the disk.
    """
    logger.info('Reading enwiki')
    if config['retriever'] in ['tfidf', 'golden']:
        logger.info('Retriever is TF-IDF/Golden method, Skips this step')
        return

    # return existing tokenized corpus
    path = get_file_path(config['enwiki_corpus_pickle'],
                         category='context_reader')
    path_tokens = get_file_path(config['enwiki_corpus_tokens_pickle'],
                               category='context_reader')
    if (os.path.isfile(path) and os.path.isfile(path_tokens)):
        logger.info('Exist tokenized enwiki file')
        return
    # initial settings
    root = config['enwiki_path']
    if 'bm25' == config['retriever']:
        tokenizer = T5Tokenizer.from_pretrained(config['bm25_tokenizer'])
    elif 'faiss' == config['retriever']:
        tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                                "facebook/dpr-ctx_encoder-multiset-base")

    # pre-reading all files for the progress bar
    file_path_list = []
    for path, dir_list, file_list in os.walk(root):
        if 0 == len(file_list):
            continue
        for file_name in file_list:
            file_path = os.path.join(path, file_name)
            file_path_list.append(file_path)

    # read all articles, each paragraph is a document
    logger.info('Total number of wiki files: ' + str(len(file_path_list)))
    manager = Manager()
    corpus_list = manager.list()
    corpus_token_list = manager.list()
    lck = Lock() # lock
    jobs = []
    count = 0

    logger.info('Tokenizing enwiki')
    pool = Pool(processes=config['n_pool_tokenize'],
                initializer=init_lock,
                initargs=(lck,))
    for file_name in tqdm(file_path_list, desc='No. of files sent to pool'):
        jobs.append(pool.apply_async(func=read_enwiki_single_file,
                                     args=(file_name,
                                           tokenizer,
                                           corpus_list,
                                           corpus_token_list)))
    pool.close()
    for job in tqdm(jobs, desc='No. of files processed'):
        count += job.get()
    logger.info('Total number of processed wiki paragraphs: ' + str(count))
    pool.join()

    if 0 != len(corpus_list):
        logger.info(str(len(corpus_list)))
        save_corpus_list = list(corpus_list)
        path = get_file_path(config['enwiki_corpus_pickle'],
                             category='context_reader')
        with open(path, 'a+b') as f:
            pickle.dump(save_corpus_list, f)
        save_corpus_token_list = list(corpus_token_list)
        path = get_file_path(config['enwiki_corpus_tokens_pickle'],
                             category='context_reader')
        with open(path, 'a+b') as f:
            pickle.dump(save_corpus_token_list, f)

    logger.info('Finish reading enwiki')

#def read_enwiki():
#    """ Read enwiki articles and return tokenized documents
#
#        corpus_token_list(2-D list):
#            each element in the outer list is a list of tokens
#            from a document. Since it is too large (RAM > 300GB)
#            to be handled by the machine (RAM < 250G), it will be
#            stored in files but not return.
#    """
#    # return existing tokenized corpus
#    if (os.path.isfile(config['enwiki_corpus_tokens_pickle']) and
#            os.path.isfile(config['enwiki_corpus_pickle'])):
#        return
#
#    # initial settings
#    root = config['enwiki_path']
#    tokenizer = T5Tokenizer.from_pretrained(config['t5'])
#
#    # pre-reading all files for the progress bar
#    file_path_list = []
#    for path, dir_list, file_list in os.walk(root):
#        if 0 == len(file_list):
#            continue
#        for file_name in file_list:
#            file_path = os.path.join(path, file_name)
#            file_path_list.append(file_path)
#
#    # read all articles, each paragraph is a document
#    corpus_list = []
#    corpus_token_list = []
#    debug_cnt = 0
#    for file_name in tqdm(file_path_list, desc='Processed wiki files'):
#        with open(file_name, 'r', encoding='utf-8') as f:
#            for line in f.readlines():
#                content = json.loads(line)
#                # remove emtpy documents
#                if 0 == len(content['text'].strip()):
#                    continue
#
#                # clean corpus
#                paragraphs = content['text'].split('\n')
#                paragraphs = [clean_string(paragraph) for paragraph in paragraphs]
#                corpus_list.extend(paragraphs)
#
#                # tokenize corpus
#                tokens_ids = tokenizer(paragraphs,
#                                       max_length=config['max_doc_length'],
#                                       truncation=True
#                                       ).input_ids
#                tokenized_paragraphs = [tokenizer.convert_ids_to_tokens(tokens_id)
#                                        for tokens_id in tokens_ids]
#                corpus_token_list.extend(tokenized_paragraphs)
#
#        # iteratively store and fetch documents, save for future BM25 use
#        if 1e4 < len(corpus_list):
#            with open(config['enwiki_corpus_pickle'], 'a+b') as f:
#                pickle.dump(corpus_list, f)
#            with open(config['enwiki_corpus_tokens_pickle'], 'a+b') as f:
#                pickle.dump(corpus_token_list, f)
#            corpus_list = []
#            corpus_token_list = []
#
#            debug_cnt += 1
#
#        # debug
#        if 2 < debug_cnt:
#            print('dooooooge')
#            break

def read_enwiki_from_mdr():
    logger.info('Reading enwiki')
    if config['retriever'] in ['tfidf', 'golden']:
        logger.info('Retriever is TF-IDF/Golden method, Skips this step')
        return

    # return existing tokenized corpus
    path = get_file_path(config['enwiki_corpus_pickle'],
                         category='context_reader')
    path_tokens = get_file_path(config['enwiki_corpus_tokens_pickle'],
                               category='context_reader')
    if (os.path.isfile(path) and os.path.isfile(path_tokens)):
        logger.info('Exist tokenized enwiki file')
        return

    with open(config['enwiki_mdr_path'], 'r') as f:
        corpus = json.load(f)
    
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(
                            "facebook/dpr-ctx_encoder-multiset-base",
                            model_max_length=config['model_max_length'])

    corpus_list = []
    corpus_token_list = []
    for idx, doc in tqdm(corpus.items()):
        text = doc['text']

        # tokenize corpus
        tokens = tokenizer(text)
        tokens = dict(tokens)
#        tokens = [dict(zip(tokens,t)) for t in zip(*tokens.values())]

        corpus_list.append(text)
        corpus_token_list.append(tokens)

    path = get_file_path(config['enwiki_corpus_pickle'],
                         category='context_reader')
    with open(path, 'a+b') as f:
        pickle.dump(corpus_list, f)
    path = get_file_path(config['enwiki_corpus_tokens_pickle'],
                         category='context_reader')
    with open(path, 'a+b') as f:
        pickle.dump(corpus_token_list, f)

    logger.info('Finish reading enwiki')

def context_reader():
    if 'mdr' == config['retriever'][-3:]:
        # for faiss_mdr
        pass
        # read_enwiki_from_mdr()
    elif 'golden' == config['retriever']:
        pass
    elif 'bm25' == config['retriever']:
        pass
    elif 'manual' == config['retriever']:
        pass
    else:
        read_enwiki_parallel()

if __name__ == '__main__':
#    read_enwiki_parallel()
    read_enwiki_from_mdr()

