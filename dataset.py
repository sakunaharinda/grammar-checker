from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from transformers import DataCollatorForSeq2Seq
import pandas as pd
from datasets import Dataset as Dset

def get_dataframe(path: str):
    return pd.read_csv(path)

class GrammarDataset(Dataset):
    def __init__(self, dataset, tokenizer, configs):         
        self.dataset = dataset
        self.pad_to_max_length = False
        self.tokenizer = tokenizer
        self.max_len = configs['max_token_length']
  
    def __len__(self):
        return len(self.dataset)


    def tokenize_data(self, example):
        input_, target_ = example['input'], example['output']

        # tokenize inputs
        tokenized_inputs = self.tokenizer(input_, pad_to_max_length=self.pad_to_max_length, 
                                            max_length=self.max_len,
                                            return_attention_mask=True)
    
        tokenized_targets = self.tokenizer(target_, pad_to_max_length=self.pad_to_max_length, 
                                            max_length=self.max_len,
                                            return_attention_mask=True)

        inputs={"input_ids": tokenized_inputs['input_ids'],
            "attention_mask": tokenized_inputs['attention_mask'],
            "labels": tokenized_targets['input_ids']
        }
        
        return inputs

  
    def __getitem__(self, index):
        inputs = self.tokenize_data(self.dataset[index])

        return inputs


def get_datasets(csv_path, tokenizer, configs, model, test_size=0.1):
    
    df = get_dataframe(csv_path)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    train_dataset = Dset.from_pandas(train_df)
    test_dataset = Dset.from_pandas(test_df)

    train_ds = GrammarDataset(train_dataset, tokenizer, configs)
    test_ds = GrammarDataset(test_dataset, tokenizer, configs)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding='longest', return_tensors='pt')

    return train_ds, test_ds, data_collator



