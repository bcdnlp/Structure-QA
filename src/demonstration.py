import json
import os
import random
import yaml
with open('src/config.yml') as f:
    config = yaml.safe_load(f)

import tiktoken
from transformers import T5Tokenizer

def token_length(text, model, tokenizer):
    if 't5' in model:
        return len(tokenizer(text).input_ids)
    elif 'openai' in model:
        return len(tokenizer.encode(text))

def graph_demo_generation(prompt, mode):
    assert mode in ['entity', 'graph']

    if 'hotpotqa' == config['dataset_name'][:8]:
        path = config['hotpotqa_graph_demo_path']
        with open(path, 'r') as f:
            all_demo = json.load(f)
    elif '2wikimultihopqa' == config['dataset_name'][:15]:
        path = config['2wikimultihopqa_graph_demo_path']
        with open(path, 'r') as f:
            all_demo = json.load(f)

    with open(path, 'r') as f:
        demo_graphs = json.load(f)

    if 't5' in config['rel_model']:
        tokenizer = T5Tokenizer.from_pretrained(
                        config['rel_model'],
                        model_max_length=config['model_max_length'])
        prompt_tokens_length = token_length(prompt, config['rel_model'],tokenizer)
        total_length = config['model_max_length']
    elif 'openai' == config['rel_model']:
        try:
            tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            tokenizer = tiktoken.get_encoding("cl100k_base")

        prompt_tokens_length = token_length(prompt, config['rel_model'], tokenizer) + 7
        total_length = config['max_total_length'] - config['max_graph_length']

    # count the lenght of prompt first
    total_length -= prompt_tokens_length

#    random.shuffle(demo_graphs)
    selected_demos = []
    for demo in demo_graphs:
        document = '\n'.join(['Document:', demo['document']])
        demo_adjust = document

        if 'direct-entity' != config['graph_type']:
            entities = '\n'.join(['Entities:', demo['entities']])
            demo_adjust = '\n\n'.join([demo_adjust, entities])

        if 'graph' == mode: # kg2 graph
            if 'useful' == config['relation']:
                question = '\n'.join(['Question:', demo['question']])
                graph = '\n'.join(['Graph:', demo['useful graph']])
                demo_adjust = '\n\n'.join([demo_adjust, question, graph])
            elif 'empty' == config['relation']:
                graph = 'Graph:\n'
                demo_adjust = '\n\n'.join([demo_adjust, graph])
            else:
                graph = '\n'.join(['Graph:', demo['graph']])
                demo_adjust = '\n\n'.join([demo_adjust, graph])

        demo_adjust += '\n\n'

        demo_tokens_length = token_length(demo_adjust, config['rel_model'], tokenizer)
        total_length -= demo_tokens_length
        # add if there are spaces
        if total_length >= 0:
            selected_demos.append(demo_adjust)
        # exit if there is no space anymore
        if total_length <= 0:
            break

    # connect all demos together
    selected_demos = ''.join(selected_demos)
    return selected_demos

def demo_generation(prompt):
    if 'hotpotqa' == config['dataset_name'][:8]:
        path = config['hotpotqa_demo_path']
        with open(path, 'r') as f:
            all_demo = json.load(f)
    elif '2wikimultihopqa' == config['dataset_name'][:15]:
        path = config['2wikimultihopqa_demo_path']
        with open(path, 'r') as f:
            all_demo = json.load(f)

    if 't5' in config['infer_model']:
        tokenizer = T5Tokenizer.from_pretrained(
                        config['infer_model'],
                        model_max_length=config['model_max_length'])
        prompt_tokens_length = token_length(prompt, config['infer_model'], tokenizer)
        total_length = config['model_max_length']
    elif 'openai' == config['infer_model']:
        try:
            tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            tokenizer = tiktoken.get_encoding("cl100k_base")

        prompt_tokens_length = token_length(prompt, config['infer_model'], tokenizer) + 7
        total_length = config['max_total_length'] - config['max_generate_length']

    # count the lenght of prompt first
    total_length -= prompt_tokens_length

    # randomly select demonstrations, try the best to add more demos
#    random.shuffle(all_demo)
    selected_demos = []
    for demo in all_demo:
        # documents
        documents = '\n'.join(['Documents:', demo['documents']])
        demo_adjust = documents

        # graph
        if 'empty' == config['graph_type']:
            graph = 'Graph:'
            demo_adjust = '\n\n'.join([demo_adjust, graph])
        elif 'none' != config['graph_type']:
            if 'none' == config['relation']:
                graph = '\n'.join(['Graph:', demo['graph1']])
            elif 'full' == config['relation']:
                graph = '\n'.join(['Graph:', demo['graph2']])
            elif 'useful' == config['relation']:
                graph = '\n'.join(['Graph:', demo['useful triples']])
            demo_adjust = '\n\n'.join([demo_adjust, graph])

        # question
        question = '\n'.join(['Question:', demo['question']])
        demo_adjust = '\n\n'.join([demo_adjust, question])

        # CoT + answer or answer only
        if config['cot']:
            answer = '\n'.join(['Answer:', demo['answer_CoT']])
        else:
            answer = '\n'.join(['Answer:', demo['answer']])
        demo_adjust = '\n\n'.join([demo_adjust, answer])

        # connection to the next demo or promt
        demo_adjust += '\n\n'

        # demo length
        demo_tokens_length = token_length(demo_adjust, config['infer_model'], tokenizer)
        total_length -= demo_tokens_length
        # add if there are spaces
        if total_length >= 0:
            selected_demos.append(demo_adjust)
        # exit if there is no space anymore
        if total_length <= 0:
            break

    # connect all demos together
    selected_demos = ''.join(selected_demos)

    return selected_demos

if __name__ == '__main__':
    print(demo_generation('doge', 'hotpotqa_dev'))

