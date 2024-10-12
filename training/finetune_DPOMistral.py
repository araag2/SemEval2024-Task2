import os
import wandb
import json
import torch
import argparse
import typing

# Util libs
from datasets.arrow_dataset import Dataset

# Model Libs
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig,DataCollatorForCompletionOnlyLM

def create_path(path : str) -> None:
    os.makedirs(path, exist_ok=True)
    assert os.path.isdir(path), f'No such dir: {path}'

def parse_args():
    parser = argparse.ArgumentParser()

    # "models/Mistral-7B-Instruct-v0.2/run_7/end_model/"
    parser.add_argument('--model_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help='model to train')
    parser.add_argument('--tokenizer_name', type=str, default="mistralai/Mistral-7B-Instruct-v0.2", help='tokenizer to use for the model')

    parser.add_argument('--checkpoint', type=str, default="models/Base-Model_Iterative-Training-Explanations_Iter-1/checkpoint-8000", help='checkpoint to load for model')
    parser.add_argument('--merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.add_argument('--no-merge', dest='merge', action='store_true', help='boolean flag to set if model is merging')
    parser.set_defaults(merge=False)

    parser.add_argument('--exp_name', type=str, default="DPO Iteration-1 Base-Model", help='Describes the conducted experiment')
    parser.add_argument('--run', type=int, default=3, help='run number for wandb logging')

    # I/O paths for models, CT, queries and qrels
    parser.add_argument('--save_dir', type=str, default="models/Base-Model_DPO_Iter-1/", help='path to model save dir')

    parser.add_argument("--train_file", default="outputs/DPO_Preferences-Train_Base-Model_Med-Mix-1000_Iter-1_2024-07-11_18-12", type=str)
    parser.add_argument("--dev_file", default="outputs/DPO_Preferences-Dev_Base-Model_Med-Mix-1000_Iter-1_2024-07-11_18-12", type=str)

    #Model Hyperparamenters
    parser.add_argument("--max_length", type=int, default=3000)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--pooling", default="mean")
    parser.add_argument("--train_epochs", default=5, type=int)
    parser.add_argument("--lr", type=float, default=2e-5)

    # Lora Hyperparameters
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=float, default=32)

    #Speed and memory optimization parameters
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_false", help="If True, use gradient checkpointing to save memory at the expense of slower backward pass.")
    args = parser.parse_args()

    return args

def create_model_and_tokenizer(args : argparse):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit= True,
        bnb_4bit_quant_type= "nf4",
        bnb_4bit_compute_dtype= torch.bfloat16,
        bnb_4bit_use_double_quant= False,
    )

    if args.merge:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config= bnb_config, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        model = PeftModel.from_pretrained(model, args.checkpoint, quantization_config= bnb_config, device_map= {"": 0}, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        model = model.merge_and_unload()
    else:
       model = AutoModelForCausalLM.from_pretrained(
            args.model_name, low_cpu_mem_usage=True,
            quantization_config= bnb_config,
            return_dict=True, torch_dtype=torch.bfloat16,
            device_map= {"": 0}
       )

    #### LLAMA STUFF
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r = args.lora_r,
        lora_alpha= args.lora_alpha,
        lora_dropout= args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"],
    )

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    return model, peft_config, tokenizer

def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    wandb.init(
        project="SemEval_Mistra",
        name = f'{args.model_name}/{args.exp_name}/run-{args.run}',
        group = f'{args.model_name}/{args.exp_name}',
        config = { arg : getattr(args, arg) for arg in vars(args)}
    )

    # Load tokenizer and model
    model, peft_config, tokenizer = create_model_and_tokenizer(args)


    train_dataset_load = json.load(open(f'{args.train_file}.json', encoding='utf8'))
    train_dataset = {"prompt" : [], "chosen" : [], "rejected" : []}

    for i in range(len(train_dataset_load["prompt"])):
        train_dataset["prompt"].append(f'{train_dataset_load["prompt"][i]}')
        train_dataset["chosen"].append(f'{train_dataset_load["chosen"][i]}')
        train_dataset["rejected"].append(f'{train_dataset_load["rejected"][i]}')
    train_dataset = Dataset.from_dict(train_dataset)


    eval_dataset_load = json.load(open(f'{args.dev_file}.json', encoding='utf8'))
    eval_dataset = {"prompt" : [], "chosen" : [], "rejected" : []}

    for i in range(len(eval_dataset_load["prompt"])):
        eval_dataset["prompt"].append(f'{eval_dataset_load["prompt"][i]}')
        eval_dataset["chosen"].append(f'{eval_dataset_load["chosen"][i]}')
        eval_dataset["rejected"].append(f'{eval_dataset_load["rejected"][i]}')
    eval_dataset = Dataset.from_dict(eval_dataset)

    training_arguments = DPOConfig(
        output_dir = args.save_dir,
        overwrite_output_dir=True,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps= 1000,
        eval_steps= 1000,
        save_total_limit= 5,
        num_train_epochs = args.train_epochs,
        per_device_train_batch_size= args.batch_size,
        optim = "paged_adamw_8bit",
        logging_steps= 1000,
        learning_rate= args.lr,
        bf16= False,
        group_by_length= True,
        lr_scheduler_type= "constant",
        load_best_model_at_end= True,
        fp16= args.fp16,
        report_to="wandb"
    )
    
    ## Data collator for completing with "Answer: YES" or "Answer: NO"
    collator = DataCollatorForCompletionOnlyLM("Answer: ", tokenizer= tokenizer)

    ## Setting sft parameters
    trainer = DPOTrainer(
        model= model,
        ref_model= None,
        data_collator= collator,
        train_dataset= train_dataset,
        eval_dataset= eval_dataset,
        peft_config= peft_config,
        tokenizer= tokenizer,
        args= training_arguments,
    )

    ## Training
    trainer.train()

    ## Save model and finish run
    create_path(f'{args.save_dir}end_model/')
    trainer.model.save_pretrained(f'{args.save_dir}end_model/')
    wandb.finish()


if __name__ == '__main__':
    main()