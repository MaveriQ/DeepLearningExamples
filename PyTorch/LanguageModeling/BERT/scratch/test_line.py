from tokenization import BasicTokenizer
from spacy.gold import align
import spacy
import os
from tqdm import tqdm

nlp = spacy.load('en_core_web_lg')
tokenizer=BasicTokenizer()


def test_line():
    line='The switches between clarity and intoxication gave me a headache, but at least the silver-haired faery’s explanation of the queens’ “gifts” helped me understand why I could want to wrap my legs around a creature who terrified me.'

    spacy_doc=nlp(line.lower())
    spacy_tokens=[str(token) for token in spacy_doc]
    spacy_tokens_pos=[token.pos_  for token in spacy_doc]
    bert_tokens=tokenizer.tokenize(line)
    diff=align(bert_tokens,spacy_tokens)[0]

    print('Spacy : {}'.format(spacy_tokens))
    print('BERT  : {}'.format(bert_tokens))
    # print('Cost  : {}'.format(diff))
    # print('POS   : {}'.format(spacy_tokens_pos))


if __name__ == "__main__":
    test_line()
