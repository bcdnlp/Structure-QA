import json
import logging
import os
import re
import time
import yaml

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)
os.environ['CUDA_VISIBLE_DEVICES'] = ', '.join([str(g) for g in config['gpu_no']])

from tqdm import tqdm

import openai
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from utils import get_file_path, clean_files
from preprocess import StructureDataset
import hotpot_evaluate
import wikimultihop_evaluate

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.FileHandler(os.path.join('log/', 'data.log'), 'w')
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def collate_fn(batch):
    if 't5' in config['infer_model']:
        batch_input_ids, batch_attn_masks, questions, answers = list(zip(*batch))

        max_length = 0
        for input_ids in batch_input_ids:
            max_length = max(max_length, len(input_ids))
#        max_length = min(max_length, config['max_length'])

        batch_input_ids = [torch.cat([torch.tensor(input_ids, dtype=torch.int),
                                      torch.zeros(max_length-len(input_ids),
                                                  dtype=torch.int)]
                                     ).unsqueeze(0)
                           for input_ids in batch_input_ids]
        batch_input_ids = torch.cat(batch_input_ids, dim=0)

        batch_attn_masks = [torch.cat([torch.tensor(attn_mask, dtype=torch.int),
                                       torch.zeros(max_length-len(attn_mask),
                                                   dtype=torch.int)]
                                      ).unsqueeze(0)
                            for attn_mask in batch_attn_masks]
        batch_attn_masks = torch.cat(batch_attn_masks, dim=0)
        
        return batch_input_ids, batch_attn_masks, questions, answers

    elif 'openai' == config['infer_model']:
        return batch

def check_metaqa(prediction, answer_all, correct_list):
    match = re.findall(r'(?=('+re.escape(answer_all)+r'))', prediction)
    if len(match):
        return True

    return False

def check_hotpotqa(prediction, answer, correct_list):
#    if 't5' in config['infer_model']:
#        return prediction == answer
#    elif 'openai' == config['infer_model']:
    prediction = prediction.lower()
    answer = answer.lower()
    match = re.findall(r'%s'%(answer), prediction)
    if len(match):
        return True
    return False
#    return None

def check_predictions(questions, predictions, answers, correct_list=[]):
    correct_answers = 0
    total_questions = 0
    if 'multi' == config['retrieval']:
        prev_question = ""
        prev_flag = False
        prev_idx = -1
    for idx, (question, prediction, answer_all) in enumerate(zip(questions,
                                                                 predictions,
                                                                 answers)):
        # check answer
        if 'metaqa' == config['dataset_name'][:6]:
            flag = check_metaqa(prediction, answer_all, correct_list)
        elif 'hotpotqa' == config['dataset_name'][:8]:
            flag = check_hotpotqa(prediction, answer_all, correct_list)

        # collect results based on retrieval method
        if 'single' == config['retrieval']:
            total_questions += 1
            if flag:
                correct_answers += 1
                if config['debug']:
                    correct_list.append(idx)
        elif 'multi' == config['retrieval']:
            if prev_question != question:
                total_questions += 1
                if prev_flag:
                    correct_answers += 1
                    if config['debug']:
                        correct_list.append(prev_idx)

                # update to current question
                prev_question = question
                prev_flag = flag
            else:
                prev_flag = prev_flag or flag
                if flag:
                    prev_idx = idx

    # count the last group of question
    if 'multi' == config['retrieval']:
        total_questions += 1
        if prev_flag:
            correct_answers += 1
            if config['debug']:
                correct_list.append(prev_idx)

    return correct_answers / total_questions

def openai_inference(dev_data):
    with open(config['openai_org_id'], 'r') as f:
        openai.organization = f.readline().strip()

    with open(config['openai_key'], 'r') as f:
        key = json.load(f)
        openai.api_key = key[config['account']].strip()

    results = []
    check_flag = True
    for batch in tqdm(dev_data, desc='Openai generation'):
        if check_flag:
            print(batch[0][1])
#            exit()
            check_flag = False
        idxes, input_text, question, answer = batch[0]
        message = {'role': 'user', 'content': input_text}
        while True:
            try:
                response = openai.ChatCompletion.create(
                                model='gpt-3.5-turbo-0301',
                                messages=[message],
                                temperature=0,
                                max_tokens=config['max_generate_length'],
                                frequency_penalty=0,
                                presence_penalty=0,
                                stop=['\n'],
                                n=1,
                                )
                break
            except Exception as e:
                print("Errrrrrrrrrrrrrrrrrrrrrrrrrrrrrrror", e)
                time.sleep(3)
#                print(response)
#                print(batch[0])
#                input()
        prediction = response['choices'][0]['message']['content']
        prediction = re.sub('\n', '', prediction)
        result = [str(idxes), question, prediction, answer]
        results.append(result)

    results.sort(key=lambda x: x[0])

    return results

def t5_inference(dev_data):
    logger.info('Loading model %s'%(config['infer_model']))
    cuda = torch.device('cuda')
    model = T5ForConditionalGeneration.from_pretrained(config['infer_model'])
    if 'inference' == config['mode']:
        model.eval()
    model.to(device=cuda)

    if 'inference' == config['mode']:
        tokenizer = T5Tokenizer.from_pretrained(config['infer_model'],
                                                model_max_length=config['model_max_length'])
        with torch.no_grad():
            results = []
            for batch in tqdm(dev_data, desc='Inferencing'):
                idxes, batch_input_ids, batch_attn_masks, questions, answers = batch
                batch_input_ids = batch_input_ids.to(device=cuda)
                batch_attn_masks = batch_attn_masks.to(device=cuda)
                
                output_sequences = model.generate(input_ids=batch_input_ids,
                                                  attention_mask=batch_attn_masks,
                                                  max_new_tokens=50,
                                                  do_sample=False)

                predictions = tokenizer.batch_decode(output_sequences,
                                                     skip_special_tokens=True)

#                # from the idea of zero-shot
#                inputs = tokenizer.batch_decode(batch_input_ids,
#                                                skip_special_tokens=True)
#                text = ['\n'.join([t, p, 'Therefore, the answer is:'])
#                        for t, p in zip(inputs, predictions)]
#                batch_inputs = tokenizer(text,
#                                         padding='longest',
#                                         return_tensors='pt')
#                batch_input_ids = batch_inputs.input_ids
#                batch_input_ids = batch_input_ids.to(device=cuda)
#                batch_attn_masks = batch_inputs.attention_mask
#                batch_attn_masks = batch_attn_masks.to(device=cuda)
#
#                output_sequences = model.generate(input_ids=batch_input_ids,
#                                                  attention_mask=batch_attn_masks,
#                                                  max_new_tokens=50,
#                                                  do_sample=False)
#
#                predictions = tokenizer.batch_decode(output_sequences,
#                                                     skip_special_tokens=True)

                # prepare for config['retrieval'] is 'multi'
                result = list(zip(idxes, questions, predictions, answers))
                results.extend(result)

    # gather same questions together for config['retrieval'] is 'multi'
    results.sort(key=lambda x: x[0])

    return results

def run():
    # remove exist files
    clean_files()

    # load data
    logger.info('Loading dataset')
    dev_dataset = StructureDataset(config['dataset_name'])
    dev_data = DataLoader(dev_dataset,
                          batch_size=config['batch_size'],
                          collate_fn=collate_fn,
                          shuffle=True)

    # inference
    logger.info('Inferencing')
    if 't5' in config['infer_model']:
        results = t5_inference(dev_data)
    elif 'openai' == config['infer_model']:
        results = openai_inference(dev_data)

    # extract answer from CoT
    if config['cot']:
        # save CoT results
        path = get_file_path(config['results_file_CoT'], category='results')
        logger.info('Writing results to file ' + path)
        with open(path, 'w', encoding='utf-8') as f:
            for result in tqdm(results):
                f.write('\t'.join(result) + '\n')

        for idx in range(len(results)):
            prediction = results[idx][2]
            extract = re.search('the answer is: (.*)\.$', prediction, re.IGNORECASE)
            if extract is not None:
                prediction = extract.group(1)
            results[idx][2] = prediction

    # save final results
    path = get_file_path(config['results_file'], category='results')
    logger.info('Writing results to file ' + path)
    with open(path, 'w', encoding='utf-8') as f:
        for result in tqdm(results):
            f.write('\t'.join(result) + '\n')

    # evaluate results
    logger.info('Evaluating')
    if 'hotpotqa' == config['dataset_name'][:8]:
        evaluate_hotpot(results)
    elif '2wikimultihopqa' == config['dataset_name'][:15]:
        evaluate_2wikimultihop(results)
    else:
        evaluate(results)

def evaluate_hotpot(results=None):
    if results is None:
        path = get_file_path(config['results_file'], category='results')
        logger.info('Loading results from file ' + path)
        results = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                result = line.split('\t')
                results.append(result)

    hotpot_predictions = {'answer': {}, 'sp': {}}
    for result in results:
        idx, question, prediction, answer = result
        hotpot_predictions['answer'][idx] = prediction

    prediction_path = get_file_path(config['results_hotpot_file'], category='results')
    with open(prediction_path, 'w') as f:
        json.dump(hotpot_predictions, f)

    dataset_file_path = config['_'.join([config['dataset_name'], 'path'])]
    hotpot_evaluate.eval(prediction_path, dataset_file_path)

def evaluate_2wikimultihop(results=None):
    if results is None:
        path = get_file_path(config['results_file'], category='results')
        logger.info('Loading results from file ' + path)
        results = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                result = line.split('\t')
                results.append(result)

    wikimultihop_predictions = {'answer': {}, 'sp': {}, 'evidence': {}}
    for result in results:
        idx, question, prediction, answer = result
        wikimultihop_predictions['answer'][idx] = prediction

    prediction_path = get_file_path(config['results_2wikimultihop_file'], category='results')
    with open(prediction_path, 'w') as f:
        json.dump(wikimultihop_predictions, f)

    dataset_file_path = config['_'.join([config['dataset_name'], 'path'])]
    id_aliases_path = config['2wikimultihopqa_id_aliases_path']
    wikimultihop_evaluate.eval(prediction_path, dataset_file_path, id_aliases_path)

def evaluate(results=None):
    if results is None:
        path = get_file_path(config['results_file'], category='results')
        logger.info('Loading results from file ' + path)
        results = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                result = line.split('\t')
                results.append(result)

    questions, predictions, answers = list(zip(*results))
    if config['debug']:
        correct_list = []
        correct_idx = []
        accuracy = check_predictions(questions,
                                     predictions,
                                     answers,
                                     correct_idx)
        for idx in correct_idx:
            correct_list.append((questions[idx],
                                 predictions[idx],
                                 answers[idx]))
    else:
        accuracy = check_predictions(questions,
                                     predictions,
                                     answers)
    

    logger.info('Accuracy: %.4f'%(accuracy))

    if config['debug']:
        print(len(correct_list))
        path = get_file_path(config['results_debug'], category='results')
        print(path)
        with open(path, 'w', encoding='utf-8') as f:
            for result in correct_list:
                f.write('\t'.join(result) + '\n')

if __name__ == '__main__':
    if not config['evaluate_only']:
        run()
    else:
        evaluate_hotpot()
#        evaluate_2wikimultihop()
#        evaluate()
