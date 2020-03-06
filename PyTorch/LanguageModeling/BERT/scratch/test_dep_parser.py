import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from spacy.tokens import Doc
import spacy
from tokenization import BasicTokenizer

def my_tokenizer(text):
    bert_tokens=basic_tokenizer.tokenize(text) 
    return Doc(nlp.vocab,words=bert_tokens)

nlp=spacy.load('en_core_web_lg')
nlp.tokenizer=my_tokenizer
never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
basic_tokenizer = BasicTokenizer(do_lower_case=True,
                                              never_split=never_split)

text='The switches between clarity and intoxication gave me a headache, but at least the silver-haired faery’s explanation of the queens’ “gifts” helped me understand why I could want to wrap my legs around a creature who terrified me.'

spacy_doc=nlp(text)

spacy_tokens=[(t.i,
            # t.text,
            # t.head.text,
            t.head.i) for t in spacy_doc]

# for token in spacy_doc:
#     print(token.text, token.dep_, token.head.text, token.head.pos_,
#             [child for child in token.children])

# for chunk in spacy_doc.noun_chunks:
#     print(chunk.text)#, chunk.root.text, chunk.root.dep_,
            #chunk.root.head.text)