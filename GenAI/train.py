import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training
from peft import LoraConfig, get_peft_model
#from accelerate import Accelerator, FullyShardedDataParallel
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
import transformers
from datetime import datetime
from datasets import load_dataset

import argparse
import os

def load_base_model():
    # Load base model
    base_model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load the model from the Hugging Face Hub
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)

    # Load the tokenizer from the Hugging Face Hub
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        padding_side='left',
        add_eos_token=True,
        add_bos_token=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    return base_model, tokenizer

def load_data():
    # Load training and evaluation datasets
    train_data_path = './data/total.json'
    eval_data_path = './data/eval.json'

    train_dataset = load_dataset('json', data_files=train_data_path, split='train')
    print(train_dataset)
    eval_dataset = load_dataset('json', data_files=eval_data_path, split='train')
    print(eval_dataset)

    tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
    tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

    return tokenized_train_dataset, tokenized_val_dataset



# For Mistral Instruct 
def formatting_func(sample):
    bos_token = "<s>"
    instruct_message = "[INST]We have 7 Level of requirement satisfaction as below:\n1. The answer is slightly relevant to the topic, but it does not really address any part of the question\nn2. The answer addresses a very small part of the question\n3. The answer only addresses part of the question. The answer has addressed a part (less than 50%) of the task’s requirement\n4. Most parts of the questions are addressed, but some parts of the answer lack focus. The answer has addressed a bigger part (50-70n%) of the task’s requirement.\n5. All parts are addressed but some parts are answered better than others. The answer has managed nto address all parts of the task’s requirement but has yet covered all the points thoroughly.\n6. All parts are addressed, most parts are answered in-depth.\nn7. The answer is so full that the marker has nothing to add.The answer has managed to address all parts of the task’s requirement and points relating to advantages and disadvantages have been covered thoroughly.[\INST]</s>"
    question = sample["question"].replace("\n\n### Question\n", "").strip()
    answer = sample["essay"].replace("\n### Essay\n", "").strip()
    score = sample["final_score"].replace("\n### Score\n", "").strip()
    eos_token = "</s>"
Score
    full_prompt = ""
    full_prompt += bos_tokenEssay
    full_prompt += instruct_message
    full_prompt += "\n" + '[INST]' + question
    full_prompt += " [/INST]\n"
    full_prompt += answer + "\n"
    full_prompt += score
    full_prompt += eos_token

    return full_prompt


def generate_and_tokenize_prompt(prompt):
    # Generate and tokenize prompts
    max_length = 1700
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def config_model(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    # Configure LoRA model
    config = LoraConfig(
    	# Use DoRA for finetuning
    	# use_dora = True, 	# Comment this if occur an error 
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,
	use_dora=True,
        task_type="CAUSAL_LM",
    )

    # Get the LoRA model
    model = get_peft_model(model, config)

    # Configure Fully Sharded Data Parallel (FSDP)
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    # Initialize the Accelerator
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
    model = accelerator.prepare_model(model)


    return model


def train_model(tokenized_train_dataset, tokenized_val_dataset, model, tokenizer):
    output_dir = './trained-model/mistral'


    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            #num_train_epochs=10,
            warmup_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
	    num_train_epochs = 10, 
            max_steps=-1,
            learning_rate=2e-5,
            fp16=True,
            optim="paged_adamw_8bit",
            logging_steps=10,
            logging_dir='./logs',
            save_strategy="steps",
            save_steps=200,
            #evaluation_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=10,
            do_eval=True,
            #report_to="wandb",		# Remove comment to use wandb 
            run_name=f"{output_dir}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        ),
        #accelerator=Accelerator(),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # Disable caching for training
    model.config.use_cache = False
    trainer.train()    


if __name__ == "__main__":
   
    # Load the base model and tokenizer
    base_model, tokenizer = load_base_model()

    # Load and preprocess the data
    tokenized_train_dataset, tokenized_val_dataset = load_data()

    # Configure the model
    model = config_model(base_model)

    # Train the model
    train_model(tokenized_train_dataset, tokenized_val_dataset, model, tokenizer) 
