from tokenization import BasicTokenizer
import spacy
from spacy.tokens import Doc

line='The switches between clarity and intoxication gave me a headache, but at least the silver-haired faery’s explanation of the queens’ “gifts” helped me understand why I could want to wrap my legs around a creature who terrified me.'

nlp=spacy.load('en_core_web_lg')
tokenizer=BasicTokenizer()

def my_tokenizer(text):
    bert_tokens=tokenizer.tokenize(text) 
    return Doc(nlp.vocab,words=bert_tokens)

nlp.tokenizer=my_tokenizer
doc=nlp(line)

print([(t.text,t.dep_,t.head.text,t.head.pos_,[c for c in t.children]) for t in doc])
