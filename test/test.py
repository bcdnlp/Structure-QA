#from transformers import T5Tokenizer, T5ForConditionalGeneration
#
#tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
#model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xl")

#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#
#tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
#model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl")
#
#input_text = "translate English to German: How old are you?"
#input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#
#outputs = model.generate(input_ids)
#print(tokenizer.decode(outputs[0]))

import json
import openai
import yaml

# constrain the use of gpus
with open('src/config.yml') as f:
    config = yaml.safe_load(f)


with open(config['openai_org_id'], 'r') as f:
    openai.organization = f.readline().strip()

with open(config['openai_key'], 'r') as f:
    keys = json.load(f)
    openai.api_key = keys['account1']

#context_p = 'The newest embeddings model is text-embedding-ada-002.'
#question_p = 'What is our newest embeddings model?'
#answer_p = 'text-embedding-ada-002'
#prompt = "Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\" Context: %s Question: %s Answer:"%(context_p, question_p)
#prompt = "Context: %s Question: %s Answer:"%(context, question)
#prompt = "context: %s\nquestion: %s\nanswer: %s\n---\ncontext: %s\nquestion: %s\nanswer:"%(context_p, question_p, answer_p, context_p, question_p)
#print(prompt)

with open('test/prompt.txt', 'r') as f:
    prompt = f.read()[:-1]

print(prompt, end='')

#                model='gpt-3.5-turbo-0301',
#                model='gpt-4-32k-0314',
#                model='gpt-4-0314',
message = {'role': 'user', 'content': prompt}
prediction = openai.ChatCompletion.create(
                model='gpt-3.5-turbo-0301',
                messages=[message],
                temperature=0,
                max_tokens=300,
                frequency_penalty=0,
                presence_penalty=0,
                stop=['\n'],
                n=1,
                )

#prediction = openai.Completion.create(prompt=prompt,
#                                      temperature=0,
#                                      max_tokens=300,
#                                      top_p=1,
#                                      frequency_penalty=0,
#                                      presence_penalty=0,
#                                      stop=['\n'],
#                                      model="gpt-4-0314",
#                                      )

#prediction = openai.Completion.create(model="code-davinci-002",
#                                      prompt="Say this is a test",
#                                      max_tokens=7,
#                                      temperature=0,
#                                      )

prediction = prediction['choices'][0]['message']['content']
#prediction = prediction['choices'][0]['text'].strip()
print(prediction)
#prediction = prediction.split('\n')[0].strip()
#print('doooooooooooooooooooooge')
#if not prediction[-1].isalnum():
#    prediction = prediction[:-1]
#print(prediction)

