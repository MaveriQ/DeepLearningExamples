#!/bin/bash                                                                                                                                                                                                                                  
start=`date +%s`
python3 /workspace/bert/data/bertPrep.py --action sharding --dataset books_wiki_en_corpus

shard1=`date +%s`
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 128 \
         --max_predictions_per_seq 20 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1 --n_processes 32

shard2=`date +%s`
python3 /workspace/bert/data/bertPrep.py --action create_hdf5_files --dataset books_wiki_en_corpus --max_seq_length 512 \
         --max_predictions_per_seq 80 --vocab_file $BERT_PREP_WORKING_DIR/download/google_pretrained_weights/uncased_L-24_H-1024_A-16/vocab.txt --do_lower_case 1 --n_processes 32
end=`date +%s`

shard_time=$((shard1-start))
hdfs1_time=$((shard2-shard1))
hdfs2_time=$((end-shard2))
total_time=$((end-start))

echo Sharding time was $shard_time seconds
echo HDFS1 time was $hdfs1_time seconds
echo HDFS2 time was $hdfs2_time seconds
echo Total time was $total_time seconds
