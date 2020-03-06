from tokenization import BertTokenizer
import random
from create_pretraining_data import create_training_instances, write_instance_to_example_file
from glob import glob
# line='The switches between clarity and intoxication gave me a headache, but at least the silver-haired faery’s explanation of the queens’ “gifts” helped me understand why I could want to wrap my legs around a creature who terrified me.'
vocab_file='/workspace/bert/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt'
base_dir='/home/haris/share/git/DeepLearningExamples/PyTorch/LanguageModeling/BERT/sharded_training_shards_256_test_shards_256_fraction_0.2/books_wiki_en_corpus/'
random_seed=123

tokenizer=BertTokenizer(vocab_file)
rng = random.Random(random_seed)
max_seq_length=128
dupe_factor=5
short_seq_prob=0.1
masked_lm_prob=0.15
max_predictions_per_seq=20
# tokens=tokenizer.tokenize(line)

# input_files=glob(base_dir+'*.txt')
input_files=['/home/haris/share/git/DeepLearningExamples/PyTorch/LanguageModeling/BERT/test_file.txt']
output_file='/home/haris/share/git/DeepLearningExamples/PyTorch/LanguageModeling/BERT/test_file.h5'

instances = create_training_instances(
        input_files, tokenizer, max_seq_length, dupe_factor,
        short_seq_prob, masked_lm_prob, max_predictions_per_seq,
        rng)

write_instance_to_example_file(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_file)