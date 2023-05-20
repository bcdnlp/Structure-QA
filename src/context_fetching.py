import json
import logging
import math
import os
import pickle
import yaml
from collections import OrderedDict

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(g) for g in config['gpu_no']])

from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.util import Finalize

import faiss
import numpy as np
import torch
import torch.nn as nn
#from rank_bm25 import BM25Okapi
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from transformers import T5Tokenizer
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import DPRContextEncoder
from transformers import AutoConfig, AutoTokenizer

from utils import clean_string, get_file_path, clean_files
from src.dataset_reader import read_data
from src.context_reader import read_enwiki_parallel

from src.drqa.retriever.tfidf_doc_ranker import TfidfDocRanker
from src.drqa.retriever.doc_db import DocDB
from src.mdr.retrieval.models.mhop_retriever import RobertaRetriever
from src.mdr.retrieval.utils.utils import load_saved, move_to_cuda
from src.elasticsearch_retriever import ElasticsearchRetriever

if config['debug']:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.DEBUG)
else:
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'context_fetching.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

for i in range(torch.cuda.device_count()):
    logger.info(torch.cuda.get_device_name(i))

#def bm25_fetching_for_query(qa_pair, corpus, tokenizer):
#    """ Fetch related paragraphs from enwiki by BM25 algorithm for a query
#
#        corpus_token_list(2-D list):
#            each element in the outer list is a list of tokens
#            from a document. Since it is too large (RAM > 300GB)
#            to be handled by the machine (RAM < 250G), it will be
#            read from a file but not be a argument.
#
#    Args:
#        qa_pair(tuple): a tuple including (question, answer)
#        corpus(list): a list of documents (possible context)
#        tokenizer: T5 tokenizer
#
#    Return:
#        context_qa_tuple(tuple): a tuple containing (context, question, answer)
#    """
#    # clean question as cleaning documents
#    question, answer = qa_pair
#    question = clean_string(question, is_question=True)
#    question_tokens_id = tokenizer(question).input_ids
#    question_tokens = tokenizer.convert_ids_to_tokens(question_tokens_id)
#
#    # run bm25 over all possible context
#    if config['top_n'] < len(corpus):
#        tokens_ids = tokenizer(corpus,
#                               max_length=config['max_doc_length'],
#                               ).input_ids
#        tokenized_corpus = [tokenizer.convert_ids_to_tokens(tokens_id)
#                            for tokens_id in tokens_ids]
#        bm25 = BM25Okapi(tokenized_corpus)
#        context = bm25.get_top_n(question_tokens, corpus, n=config['top_n'])
#    else:
#        context = corpus
#
#    context_qa_tuple = (context, question, answer)
#
#    return context_qa_tuple
#
#def bm25_for_chunk(qa_pairs, corpus, corpus_tokens, tokenizer):
#    """
#        qa_pairs in order.
#    """
#    logger.info('corpus length: ' + str(len(corpus)))
#    logger.info('corpus token length: ' + str(len(corpus_tokens)))
#    logger.info('corpus first line: ' + corpus[0])
#    bm25 = BM25Okapi(corpus_tokens)
#    chunk_context = []
#    for qa_pair in qa_pairs:
#        # clean question as cleaning documents
#        question, answer = qa_pair
#        question = clean_string(question, is_question=True)
#        question_tokens_id = tokenizer(question).input_ids
#        question_tokens = tokenizer.convert_ids_to_tokens(question_tokens_id)
#
#        context = bm25.get_top_n(question_tokens, corpus, n=config['top_n'])
#        chunk_context.append(context)
#
#    return chunk_context
#
#def bm25_fetching(qa_pairs, dataset_name):
#    """ Fetch related paragraphs from enwiki by BM25 algorithm
#        corpus_token_list(2-D list):
#            each element in the outer list is a list of tokens
#            from a document. Since it is too large (RAM > 300GB)
#            to be handled by the machine (RAM < 250G), it will be
#            read from a file but not be a argument.
#
#    Args:
#        qa_pairs(list of tuple): each element in the list is a tuple including
#                                 (question, answer)
#        dataset_name(str): a string indicating which file to find and generate
#
#    Return:
#        context_qa_tuples(list of tuple): each element in the list is a tuple
#                                          including (context, question, answer)
#    """
#
#    logger.info('Start BM25 fetching')
#
#    # return existing context_qa_tuples
#    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
#                              category='fetching_results')
#    if os.path.isfile(file_path):
#        with open(file_path, 'rb') as f:
#            context_qa_tuples = pickle.load(f)
#
#        logger.info('Finish BM25 fetching')
#
#        return context_qa_tuples
#
#    # initial tokenizer
#    tokenizer = T5Tokenizer.from_pretrained(config['bm25_tokenizer'],
#                                            model_max_length=config['model_max_length'])
#
#    # iteratively build BM25 for each chunk
#    possible_context = []
#    jobs = []
#    pool = Pool(processes=config['n_pool_bm25'])
#    path = get_file_path(config['enwiki_corpus_pickle'],
#                         category='fetching_results')
#    path_tokens = get_file_path(config['enwiki_corpus_tokens_pickle'],
#                               category='fetching_results')
#    with open(path, 'rb') as f_corpus, \
#            open(path_tokens, 'rb') as f_corpus_tokens:
#        try:
#            while True:
#                corpus = pickle.load(f_corpus)
#                corpus_tokens = pickle.load(f_corpus_tokens)
#                chunk_context_handler = pool.apply_async(func=bm25_for_chunk,
#                                                         args=(qa_pairs,
#                                                               corpus,
#                                                               corpus_tokens,
#                                                               tokenizer))
#                jobs.append(chunk_context_handler)
#                if config['debug']:
#                    if 1 < len(jobs):
#                        break
#        except EOFError:
#            pass
#
#    pool.close()
#    logger.info(str(len(jobs)))
#    for chunk_context_handler in tqdm(jobs, desc='Processed chunk'):
#        possible_context.append(chunk_context_handler.get())
#    pool.join()
#
#    # convert list of context in a single chunk to list of context for a question
#    corpus = list(zip(*possible_context_handler))
#
#    # build BM25 for each question and fetch the top n related documents
#    pool = Pool(processes=config['n_pool_bm25'])
#    jobs = []
#    context_qa_tuples = []
#    for qa_pair, possible_context in zip(qa_pairs, corpus):
#        context_handler = pool.apply_async(func=bm25_fetching_for_query,
#                                           args=(qa_pair,
#                                                 possible_context,
#                                                 tokenizer))
#        jobs.append(context_handler)
#    pool.close()
#    for context_handler in tqdm(jobs, desc='Processed question:'):
#        context_qa_tuples.append(context_handler.get())
#    pool.join()
#
#    # save for future use
#    with open(file_path, 'wb') as f:
#        pickle.dump(context_qa_tuples, f)
#
#    logger.info('Finish BM25 fetching')
#
#    return context_qa_tuples

def elasticsearch_bm25_fetching(qa_pairs, dataset_name):
    logger.info("Start bm25 fetching")

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_qa_tuples = pickle.load(f)

        logger.info('Load bm25 Fetching Result')

        return context_qa_tuples

    retriever = ElasticsearchRetriever(
        corpus_name=config['dataset'],
        elasticsearch_host=config['elastic_host'],
        elasticsearch_port=config['elastic_port'],
    )

    idxes, contexts, questions, answers = [], [], [], []
    for qa_pair in tqdm(qa_pairs):
        idx, question, answer = qa_pair
        idxes.append(idx)
        questions.append(question)
        answers.append(answer)

        results = retriever.retrieve_paragraphs(
                        question,
                        max_hits_count=config['top_n'])
        for idx in range(len(results)):
            result = results[idx]
            title = ' '.join(['Wikipedia Title:', result['title']])
            document = result['paragraph_text']
            results[idx] = '\n'.join([title, document])
#        results = [result['paragraph_text'] for result in results]
        context = '\n\n'.join(results)
        contexts.append(context)

    context_qa_tuples = list(zip(idxes, contexts, questions, answers))

    logger.info('Saving Results')
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_qa_tuples, f)

    logger.info("Finish bm25 fetching")

    return context_qa_tuples

def dpr_faiss_fetching(qa_pairs, dataset_name):
    """ Fetch related paragraphs from enwiki by DPR and FAISS  algorithm
        corpus_token_list(2-D list):
            each element in the outer list is a list of tokens
            from a document. Since it is too large (RAM > 300GB)
            to be handled by the machine (RAM < 250G), it will be
            read from a file but not be a argument.

    Args:
        qa_pairs(list of tuple): each element in the list is a tuple including
                                 (question, answer)
        dataset_name(str): a string indicating which file to find and generate

    Return:
        context_qa_tuples(list of tuple): each element in the list is a tuple
                                          including (context, question, answer)
    """

    logger.info('Start FAISS fetching')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_qa_tuples = pickle.load(f)

        logger.info('Load existing FAISS fetching results')

        return context_qa_tuples

    # initial tokenizer and embedding models
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
                            "facebook/dpr-question_encoder-multiset-base")
    question_model = DPRQuestionEncoder.from_pretrained(
                            "facebook/dpr-question_encoder-multiset-base")
    context_model = DPRContextEncoder.from_pretrained(
                            "facebook/dpr-ctx_encoder-multiset-base")
    context_model.eval()
    cuda = torch.device('cuda')
    if 1 < len(config['gpu_no']):
        question_model = nn.DataParallel(question_model)
        context_model = nn.DataParallel(context_model)
    question_model.to(device=cuda)
    context_model.to(device=cuda)

    # embed context for faiss index
    embeddings_list = []
    count = 0
    path_dpr = get_file_path(config['context_dpr_embedding_bin'],
                             category='context_index')
    if os.path.isfile(path_dpr):
        logger.info('Reading context DPR embeddings')
        with open(path_dpr, 'rb') as f_dpr_embedding:
            try:
                while True:
                    if 0 == count % 1000:
                        logger.info('Reading chunk %d'%(count))
                    embeddings = np.load(f_dpr_embedding)
                    embeddings_list.append(embeddings)
                    count += 1
            except ValueError:
                pass
    else:
        logger.info('Embedding context DPR')
        path_tokens = get_file_path(config['enwiki_corpus_tokens_pickle'],
                                    category='context_reader')
        with open(path_tokens, 'rb') as f_corpus_tokens, \
                open(path_dpr, 'wb') as f_dpr_embedding:
            try:
                while True:
                    corpus_tokens = pickle.load(f_corpus_tokens)
                    for i in tqdm(range(math.ceil(len(corpus_tokens)/config['batch_size']))):
                        count += min(config['batch_size'],
                                     len(corpus_tokens) - i*config['batch_size'])
                        # build a batch
                        context_list = corpus_tokens[i*config['batch_size']:
                                                     (i+1)*config['batch_size']]

                        max_length = 0
                        for context in context_list:
                            max_length = max(max_length, len(context['input_ids']))
                        max_length = min(max_length, config['max_length'])

                        # pack input_ids batch
                        tokens_ids = [torch.tensor(context['input_ids'],
                                                   dtype=torch.int)
                                        for context in context_list]
                        tokens_ids = [torch.cat([tokens_id, \
                                                 torch.zeros(max_length-len(tokens_id),
                                                             dtype=torch.int)]
                                                             ).unsqueeze(0)
                                        for tokens_id in tokens_ids]
                        tokens_ids = torch.cat(tokens_ids, dim=0)
                        tokens_ids = tokens_ids.to(device=cuda)

                        # pack attention_mask batch
                        attn_masks = [torch.tensor(context['attention_mask'],
                                                   dtype=torch.int)
                                        for context in context_list]
                        attn_masks = [torch.cat([attn_mask,
                                                 torch.zeros(max_length-len(attn_mask),
                                                             dtype=torch.int)]
                                                             ).unsqueeze(0)
                                        for attn_mask in attn_masks]
                        attn_masks = torch.cat(attn_masks, dim=0)
                        attn_masks = attn_masks.to(device=cuda)

                        # calculate embeddings
                        with torch.no_grad():
                            embeddings = context_model(input_ids=tokens_ids,
                                                       attention_mask=attn_masks).pooler_output
                            embeddings = embeddings.cpu().detach().numpy()
                        np.save(f_dpr_embedding, embeddings)
                        embeddings_list.append(embeddings)

                        if config['debug']:
                            break
                    logger.info('Embeded %d context'%(count))
                    if config['debug']:
                        break
            except EOFError:
                pass
    if 1 < len(embeddings_list):
        embeddings_list = np.concatenate(embeddings_list, axis=0)

    # build faiss index and move to GPU
    logger.info('Building FAISS index')
#    res = faiss.StandardGpuResources()
#    if config['debug']:
#        index = faiss.index_factory(embeddings_list.shape[-1], "PCA64,IVF8_HNSW32,Flat")
#    else:
#        index = faiss.index_factory(embeddings_list.shape[-1], "PCA64,IVF65536_HNSW32,Flat")
#    index_ivf = faiss.extract_index_ivf(index)
#    clustering_index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(index_ivf.d))
    index = faiss.IndexFlatL2(embeddings_list.shape[-1])
    clustering_index = faiss.index_cpu_to_all_gpus(index)
#    clustering_index = faiss.index_cpu_to_gpu(res,
#                                              config['gpu_no'],
#                                              faiss.IndexFlatL2(index_ivf.d)) # move to gpu
#    index_ivf.clustering_index = clustering_index
    logger.info('Training faiss index')
    index.train(embeddings_list)
    index.add(embeddings_list)

    # embed questions for querying
    logger.info('Embedding questions')
    question_embedding_list = []
    for qa_pair in qa_pairs:
        # clean question as cleaning documents
        question, answer = qa_pair
        question = clean_string(question, is_question=True)
        question_tokens_id = question_tokenizer(question, return_tensors='pt').input_ids
        question_tokens_id = question_tokens_id.to(cuda)
        question_embedding = question_model(question_tokens_id).pooler_output
        question_embedding_list.append(question_embedding.cpu().detach().numpy()) # TODO: check size
    question_embedding_list = np.concatenate(question_embedding_list, axis=0)

    # fetch context for each question
    logger.info('Searching for top n paragraphs')
    _, ids = index.search(question_embedding_list, config['top_n'])
    for idx in ids:
        for i in idx:
            if i == -1:
                print(idx)
                input()

    # read document text based on ids
    logger.info('Converting ids to texts')
    with open(config['enwiki_mdr_path'], 'r') as f:
        id2doc = json.load(f)

    contexts = [[id2doc[str(i)]['text'] for i in top_n_ids] for top_n_ids in ids]

    # build tuples
    logger.info('Combining context with questions and answers')
    questions, answers = list(zip(*qa_pairs))
    context_qa_tuples = list(zip(contexts, questions, answers))

    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_qa_tuples, f)

    logger.info('Finish FAISS fetching')

    return context_qa_tuples

PROCESS_DB = None

def init_db(db):
    global PROCESS_DB
    PROCESS_DB = db
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)

def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

def tfidf_fetching(qa_pairs, dataset_name):
    """ Fetch related paragraphs from enwiki by DPR and FAISS  algorithm
        corpus_token_list(2-D list):
            each element in the outer list is a list of tokens
            from a document. Since it is too large (RAM > 300GB)
            to be handled by the machine (RAM < 250G), it will be
            read from a file but not be a argument.

    Args:
        qa_pairs(list of tuple): each element in the list is a tuple including
                                 (question, answer)
        dataset_name(str): a string indicating which file to find and generate

    Return:
        context_qa_tuples(list of tuple): each element in the list is a tuple
                                          including (context, question, answer)
    """

    logger.info('Start TF-IDF Fetching')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_qa_tuples = pickle.load(f)

        logger.info('Load TF-IDF Fetching Result')

        return context_qa_tuples

    # fetch context for each question
    logger.info('Start Reading Questions and Fetching Context')
    questions = []
    answers = []
    path = get_file_path(config['tfidf_docids_path'],
                         category='context_index')
    if os.path.isfile(path):
        for qa_pair in qa_pairs:
            # clean question as cleaning documents
            question, answer = qa_pair
            question = clean_string(question, is_question=True)
            questions.append(question)
            answers.append(answer)

        logger.info('Load ids of fetched documents')
        with open(path, 'rb') as f:
            all_docids = pickle.load(f)
    else:
        logger.info('Fetching ids of top n related documents')
        path = get_file_path(config['tfidf_model_path'],
                             category='context_index')
        ranker = TfidfDocRanker(path)
        all_docids = []
        for qa_pair in tqdm(qa_pairs):
            # clean question as cleaning documents
            question, answer = qa_pair
            question = clean_string(question, is_question=True)
            questions.append(question)
            answers.append(answer)

            docids, _ = ranker.closest_docs(question, k=config['top_n'])
            all_docids.append(docids)

        with open(path, 'wb') as f:
            pickle.dump(all_docids, f)

    # fetch text according to ids
    logger.info('Converting ids to texts')
    path = get_file_path(config['tfidf_db_path'],
                         category='context_index')
    db = DocDB(path)
    flat_docids = list({d for docids in all_docids for d in docids}) # set
    pool = Pool(processes=config['n_pool_tfidf'],
                initializer=init_db,
                initargs=(db,))
    doc_texts = pool.map(fetch_text, flat_docids)
    pool.close()
    pool.join()
    doc_texts = [txt.split('\n')[0] for txt in doc_texts] # grab the first paragraph
    did_context_dict = {did:txt for did, txt in zip(flat_docids, doc_texts)}

    # zip together
    contexts = []
    for docids in all_docids:
        context = []
        for d in docids:
            context.append(did_context_dict[d])
        contexts.append(context)
    context_qa_tuples = list(zip(contexts, questions, answers))

    logger.info('Saving Results')
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_qa_tuples, f)

    logger.info('Finish TF-IDF fetching')

    return context_qa_tuples

def mdr_fetching(qa_pairs, dataset_name):
    logger.info("Start MDR fetching")

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_qa_tuples = pickle.load(f)

        logger.info('Load MDR Fetching Result')

        return context_qa_tuples

    logger.info("Initializing retrieval module...")
    bert_config = AutoConfig.from_pretrained('models/roberta-transformer-2.11/')
    tokenizer = AutoTokenizer.from_pretrained('models/roberta-transformer-2.11/')
    retriever = RobertaRetriever(bert_config, 'models/roberta-transformer-2.11/')
    retriever = load_saved(retriever, 'models/MDR/q_encoder.pt', exact=False)
    if 1 < len(config['gpu_no']):
        retriever = nn.DataParallel(retriever)
    cuda = torch.device('cuda')
    retriever.to(cuda)
    retriever.eval()

    logger.info("Loading index...")
    path = get_file_path(config['context_mdr_embedding_bin'],
                         category='context_index')
    xb = np.load(path).astype('float32')
    index = faiss.IndexFlatIP(xb.shape[-1])
    index.add(xb)
    if 1 < len(config['gpu_no']):
        index = faiss.index_cpu_to_all_gpus(index)
    else:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, config['gpu_no'][0], index)

    logger.info("Loading documents...")
    path = get_file_path(config['context_mdr_id2doc_bin'],
                         category='context_index')
    with open(path, 'r') as f:
        id2doc = json.load(f)

    logger.info("Index ready...")

    logger.info("Retrieving")
    with torch.no_grad():
        questions, answers = [], []
        for qa_pair in qa_pairs:
            question, answer = qa_pair
            questions.append(question)
            answers.append(answer)

        # first hop
        q_embeddings = []
        for i in tqdm(range(0, len(questions), config['batch_size_mdr']),
                      desc='1st-hop'):
            questions_batch = questions[i:i+config['batch_size_mdr']]
            q_encodes = tokenizer(questions_batch,
                                  max_length=config['max_doc_length'],
                                  padding='longest',
                                  truncation=True,
                                  return_tensors="pt")
            q_encodes = move_to_cuda(dict(q_encodes))
            q_embeds = retriever(q_encodes["input_ids"],
                                 q_encodes["attention_mask"],
                                 q_encodes.get("token_type_ids", None)
                                 ).cpu().numpy()
            q_embeddings.append(q_embeds)
        q_embeddings = np.concatenate(q_embeddings, axis=0)
        scores_1, docid_1 = index.search(q_embeddings, config['top_n'])

        # second hop
        query_pairs = []
        for p_idx, (question, docids) in enumerate(zip(questions, docid_1)):
            for d_idx, idx in enumerate(docids):
                doc = id2doc[str(idx)]['text']
                if '' == doc.strip():
                    doc = id2doc[str(idx)]['title']
                    scores_1[p_idx][d_idx] = float('-inf')
                query_pairs.append((question, doc))

        q_sp_embeddings = []
        for i in tqdm(range(0, len(query_pairs), config['batch_size_mdr']),
                      desc='2nd-hop'):
            query_pairs_batch = query_pairs[i:i+config['batch_size_mdr']]
            q_sp_encodes = tokenizer(query_pairs_batch,
                                     max_length=config['max_doc_length'],
                                     padding='longest',
                                     truncation=True,
                                     return_tensors="pt")
            q_sp_encodes = move_to_cuda(dict(q_sp_encodes))
            q_sp_embeds = retriever(q_sp_encodes["input_ids"],
                                    q_sp_encodes["attention_mask"],
                                    q_sp_encodes.get("token_type_ids", None)
                                    ).cpu().numpy()
            q_sp_embeddings.append(q_sp_embeds)
        q_sp_embeddings = np.concatenate(q_sp_embeddings, axis=0)
        scores_2, docid_2 = index.search(q_sp_embeddings, config['top_n'])

    # re-rank
    scores_2 = scores_2.reshape(len(questions), config['top_n'], config['top_n'])
    docid_2 = docid_2.reshape(len(questions), config['top_n'], config['top_n'])
    path_scores = np.expand_dims(scores_1, axis=2) + scores_2
    sort_results = np.flip(np.argsort(path_scores.reshape((len(questions), -1)),
                                      axis=1)
                           )[:, :config['top_n']]

    sort_results = sort_results.tolist()
    # extract context text
    contexts = []
    for q_idx, results in enumerate(sort_results):
        context = []
        for idx in results:
            x, y = idx // config['top_n'], idx % config['top_n']
            doc1_id = str(docid_1[q_idx, x])
            doc2_id = str(docid_2[q_idx, x, y])
            text = '\n\n'.join([id2doc[doc1_id]['text'], id2doc[doc2_id]['text']])
            context.append(text)
        contexts.extend(context)
#        contexts.extend(context[:1]) # for openai test

    # each question has multiple context combination, try all of them and select the best one
    if 'multi' == config['retrieval']:
        questions = [question for question in questions for _ in range(config['top_n'])]
        answers = [answer for answer in answers for _ in range(config['top_n'])]
    context_qa_tuples = list(zip(contexts, questions, answers))

    logger.info('Saving Results')
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_qa_tuples, f)

    logger.info("Finish MDR fetching")

    return context_qa_tuples

def golden_fetching(qa_pairs, dataset_name):
    ''' Test the implementation of t5 by using the hotpotqa's golden data in distractor setting
        read hotpot_dev_distractor_v1.json
    '''
    logger.info('Start GOLDEN Fetching')

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_qa_tuples = pickle.load(f)

        logger.info('Load GOLDEN Fetching Result')

        return context_qa_tuples

    path = config['_'.join([dataset_name, 'path'])]
    with open(path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    context_qa_tuples = []
    error_count = 0
    for data in all_data:
        context_options = data['context']
        context_options = {context[0]:context[1] for context in context_options}
        supporting_facts = data['supporting_facts']
        supporting_paras = list(OrderedDict.fromkeys([fact[0] for fact in supporting_facts]))
        try:
            context = ['\n'.join([': '.join(['Wikipedia Title', para_idx]),
                                  ' '.join(context_options[para_idx])])
                        for para_idx in supporting_paras]
            context = '\n'.join(context)
            context_qa_tuples.append((data['_id'], context, data['question'], data['answer']))
        except IndexError:
            error_count += 1

    logger.info('Saving Results to ' + file_path)
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_qa_tuples, f)

    return context_qa_tuples

def manual_fetching(qa_pairs, dataset_name):
    logger.info("Start manual fetching")

    # return existing context_qa_tuples
    file_path = get_file_path(config['context_qa_tuple_text_pickle'],
                              category='fetching_results')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            context_qa_tuples = pickle.load(f)

        logger.info('Load manual Fetching Result')

        return context_qa_tuples

    path = config['_'.join([dataset_name, 'path'])]

    """
    _, suffix = os.path.splitext(path)
    if '.txt' == suffix:
        with open(path, 'r', encoding='utf-8') as f:
            all_data = f.read().split('\n----\n')

        context_qa_tuples = []
        for data in all_data:
            p1, p2, question, answer = data.split('\n\n')
            context = '\n\n'.join([p1, p2])
            context_qa_tuples.append([context, question, answer])
    elif '.json' == suffix:
    """

    with open(path, 'r') as f:
        all_data = json.load(f)

    context_qa_tuples = []
    for data in all_data:
        context_qa_tuples.append([data['_id'], data['documents'], data['question'], data['answer']])

    logger.info('Saving Results')
    # save for future use
    with open(file_path, 'wb') as f:
        pickle.dump(context_qa_tuples, f)

    logger.info("Finish manual fetching")

    return context_qa_tuples

def context_fetching(qa_pairs):
    if 'bm25' == config['retriever']:
        context_qa_tuples_text = elasticsearch_bm25_fetching(qa_pairs,
                                                             config['dataset_name'])
    elif 'faiss' == config['retriever']:
        context_qa_tuples_text = dpr_faiss_fetching(qa_pairs, config['dataset_name'])
    elif 'faiss_mdr' == config['retriever']:
        context_qa_tuples_text = mdr_fetching(qa_pairs, config['dataset_name'])
    elif 'tfidf' == config['retriever']:
        context_qa_tuples_text = tfidf_fetching(qa_pairs, config['dataset_name'])
    elif 'golden' == config['retriever']:
        context_qa_tuples_text = golden_fetching(qa_pairs, config['dataset_name'])
    elif 'manual' == config['retriever']:
        context_qa_tuples_text = manual_fetching(qa_pairs, config['dataset_name'])
    
    return context_qa_tuples_text
    

if __name__ == '__main__':
    clean_files()
#    read_enwiki_parallel()
    qa_pairs = read_data(config['dataset_name'])
    context_qa_tuples = context_fetching(qa_pairs)

