import spacy

nlp = spacy.load("en_core_web_trf")#, enable=['tok2vec', 'ner'])

text = []
with open('text', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        if 0 == len(line):
            continue
        text.append(line)

print(text[0])
doc = nlp(text[0])
print(doc.ents)
print()
print(text[1])
doc = nlp(text[1])
print(doc.ents)

