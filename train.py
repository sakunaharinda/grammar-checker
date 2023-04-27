import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

import wandb

from transformers import (
    T5ForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)

from config_loader import get_configs
from metric import compute_metrics
from dataset import get_datasets
from tokenizer import get_tokenizer



if __name__ == '__main__':

    configs = get_configs('config.yaml')

    print("\nSetting up wandb ...")
    wandb.login(key=os.environ.get('wandbKey'))
    run = wandb.init(project="Grammar Checker", entity="sakuna", save_code = True)
    

    tokenizer = get_tokenizer()
    model = T5ForConditionalGeneration.from_pretrained(configs['model'])
    train_ds, test_ds, data_collator = get_datasets('c4-200m-1m.csv', tokenizer, configs, model, test_size=0.1)

    # print(train_ds[0])

    print(f"Datasets are ready! :: Train samples: {len(train_ds)} :: Test samples: {len(test_ds)}")

    args = Seq2SeqTrainingArguments(output_dir=configs['output_dir'],
                        evaluation_strategy=configs['evaluation_strategy'],
                        per_device_train_batch_size=configs['per_device_train_batch_size'],
                        per_device_eval_batch_size=configs['per_device_eval_batch_size'],
                        learning_rate=configs['learning_rate'],
                        num_train_epochs=configs['epochs'],
                        weight_decay=configs['weight_decay'],
                        save_strategy=configs['save_strategy'],
                        predict_with_generate=configs['predict_with_generate'],
                        fp16 = configs['fp16'],
                        gradient_accumulation_steps = configs['gradient_accumulation_steps'],
                        eval_steps = configs['eval_steps'],
                        save_steps = configs['save_steps'],
                        load_best_model_at_end=configs['load_best_model_at_end'],
                        logging_dir=configs['logging_dir'],
                        report_to=configs['report_to'])

    print("Start Training ...\n")

    trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                train_dataset= train_ds,
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics)

    trainer.train() 


    
    


