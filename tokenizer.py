from transformers import T5Tokenizer
from config_loader import get_configs

def get_tokenizer() -> T5Tokenizer:
    configs = get_configs('config.yaml')
    tokenizer = T5Tokenizer.from_pretrained(configs['model'])
    return tokenizer
