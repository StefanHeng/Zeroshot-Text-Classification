from transformers import GPT2TokenizerFast, GPT2ForSequenceClassification

from zeroshot_classifier.models.gpt2 import MODEL_NAME, HF_MODEL_NAME

MODEL_NAME = f'Explicit Pretrain Aspect {MODEL_NAME}'
HF_MODEL_NAME = 'bert-base-uncased'


if __name__ == '__main__':
    from stefutil import *

    tokenizer = GPT2TokenizerFast.from_pretrained(HF_MODEL_NAME)
    model = GPT2ForSequenceClassification.from_pretrained(HF_MODEL_NAME)
    # Include `end-of-turn` token for sgd, cannot set `eos` for '<|endoftext|>' already defined in GPT2
    mic(tokenizer)
    mic(tokenizer.eos_token)
    mic(type(model))
