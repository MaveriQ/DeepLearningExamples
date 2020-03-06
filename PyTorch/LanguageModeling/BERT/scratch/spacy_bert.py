from tokenization import BertTokenizer, BasicTokenizer
import spacy

from spacy.gold import align


nlp = spacy.load('en_core_web_lg')
nlp_bert = spacy.load("en_trf_bertbaseuncased_lg")

text=open('sharded_training_shards_256_test_shards_256_fraction_0.2/books_wiki_en_corpus/books_wiki_en_corpus_training_1.txt').readlines()

doc=nlp(text[0])

print(doc._.trf_word_pieces_)