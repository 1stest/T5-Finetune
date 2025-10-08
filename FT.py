###########
# Imports #
###########
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

#################
# Loading Model #
#################
model_name = "T5_Large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.to('cuda')
model.gradient_checkpointing_enable()

###########
# Padding #
###########
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=16, 
    return_tensors='pt'
    )

training_args = Seq2SeqTrainingArguments(
    output_dir="checkpoints/",
    overwrite_output_dir=True,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=500,
    label_smoothing_factor=0.1,

    fp16=True,

    logging_dir="logs/",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,

    predict_with_generate=True,
    generation_max_length=1024,
    generation_num_beams=4
)

################################################
#  IRP Pair Loading + Training Loop Asignation #
################################################
Dataset = load_from_disk("TokenizedPairs/")

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=Dataset["train"],
    eval_dataset=Dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)


##################
# Traaining Loop #
##################
trainer.train()
model.save_pretrained("lora-adapter/")


##########################
# ZeDev #  08/ 10/ 2025  #
##########################
