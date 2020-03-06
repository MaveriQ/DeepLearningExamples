from tokenization import BertTokenizer
from spacy.gold import align
import spacy
import os
from tqdm import tqdm
# from spacy.tokenizer import Tokenizer

nlp = spacy.load('en_core_web_lg')
tokenizer_bert=BertTokenizer()
# nlp.tokenier=my_basic_tokenizer
# tokenizer_spacy= Tokenizer(nlp.vocab) 

base_dir='/home/haris/share/git/DeepLearningExamples/PyTorch/LanguageModeling/BERT/sharded_training_shards_256_test_shards_256_fraction_0.2/books_wiki_en_corpus/'

# def my_basic_tokenizer(text):
#     bert_tokens=tokenizer_bert.tokenize(text)
#     return Doc(nlp.vocab,words=bert_tokens)

def test_file(file_number):
    filename='books_wiki_en_corpus_'+'training_'+str(file_number)+'.txt'
    file=os.path.join(base_dir,filename)

    total_diffs=0
    with open(file) as f:
        text=f.readlines()

        for line in text:
            
            # spacy_doc=nlp(line.lower())
            # spacy_tokens=[str(token) for token in spacy_doc]

            bert_tokens=tokenizer_bert.tokenize(line)
            doc=Doc(nlp.vocab,words=bert_tokens)
            spacy_tokens=[t.text for t in doc]
            
            diff=align(bert_tokens,spacy_tokens)[0]

            if diff!=0:
                print('Difference of '+str(diff)+' positions at line : '+line+'\n')
                total_diffs+=1
    
    print('Total number of files with differences : {}'.format(total_diffs))

if __name__ == "__main__":
    test_file(1)
