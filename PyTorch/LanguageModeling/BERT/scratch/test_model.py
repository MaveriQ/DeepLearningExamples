from modeling import BertForPreTraining, BertConfig
from torch.utils.data import DataLoader, RandomSampler 
from run_pretraining import pretraining_dataset


dataset = pretraining_dataset('test_file.h5',100)   
train_sampler = RandomSampler(dataset)
train_dataloader = DataLoader(dataset, sampler=train_sampler,batch_size=32,num_workers=8, pin_memory=True)

 
config=BertConfig(30522)
model=BertForPreTraining(config).cuda()

batch=next(iter(train_dataloader))
batch=[data.cuda() for data in batch]

input_ids, tag_ids, segment_ids, input_mask, masked_lm_labels, masked_lm_tags, next_sentence_labels = batch

loss = model(input_ids=input_ids, tag_ids=tag_ids, token_type_ids=segment_ids, attention_mask=input_mask,
            masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_labels)

print(loss)
