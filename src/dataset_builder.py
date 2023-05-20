import json
import re
import yaml
# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)

from tqdm import tqdm
from elasticsearch_retriever import ElasticsearchRetriever

if __name__ == '__main__':
    retriever = ElasticsearchRetriever(
        corpus_name=config['dataset'],
        elasticsearch_host=config['elastic_host'],
        elasticsearch_port=config['elastic_port'],
    )

    file_path = config['_'.join(['hotpotqa_dev', 'path'])]
    with open(file_path, 'r', encoding='utf-8') as f:
        all_data = json.load(f)

    new_data = []
    for data in tqdm(all_data):
        question = data['question']
        answer = data['answer']

        results = retriever.retrieve_paragraphs(
                        question,
                        max_hits_count=config['top_n'])

        titles = [result['title'] for result in results]
        documents = [result['paragraph_text'] for result in results]

        for document in documents:
            match = re.findall(r'(?=('+re.escape(answer.lower())+r'))', document.lower())
            if len(match):
                new_data.append(data)
                break

    print(len(new_data))
#    for data in new_data:
#        print(data['question'], data['answer'])

    path = f'data/hotpotqa/new_dataset_{config["top_n"]}.json'
    with open(path, 'w') as f:
        json.dump(new_data, f)

