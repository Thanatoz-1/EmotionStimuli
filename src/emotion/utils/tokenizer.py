__author__ = "Tushar Dhyani"
# File for containing all the tokenizers of this project.

from transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
