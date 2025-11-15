import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.
    text = example['text']
    
    def synonym_rep(text, n=10):
        words = text.split()
        new_words = words.copy()
        random_word_list = list(set([w for w in words if len(wordnet.synsets(w)) > 0]))
        random.shuffle(random_word_list)
        num_replaced = 0
    
        for random_word in random_word_list:
            synonyms = wordnet.synsets(random_word)
            if not synonyms:
                continue
            synonym_words = [
                lemma.name().replace('_', ' ')
                for syn in synonyms
                for lemma in syn.lemmas()
                if lemma.name().lower() != random_word.lower()
            ]
            if synonym_words:
                synonym = random.choice(synonym_words)
                new_words = [synonym if w == random_word else w for w in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
    
        return " ".join(new_words)
    
    
    def random_typo(text, prob=0.05):
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < prob and chars[i].isalpha():
                chars[i] = random.choice('qwertyuiopasdfghjklzxcvbnm,.;1234567890')
        return ''.join(chars)
    
    def random_insert(text, n=3):
        words = text.split()
        for _ in range(n):
            new_word = random.choice(words)
            pos = random.randint(0, len(words)-1)
            words.insert(pos, new_word)
        return " ".join(words)
    
    
    def random_deletion(text, p=0.1):
        words = text.split()
        new_words = [w for w in words if random.random() > p]
        return " ".join(new_words)
    
    def random_swap(text, n=3):
        words = text.split()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return " ".join(words)

    choice = random.choice(["syn","typo","del","add","swap"])
    
    if choice == "syn":
        text = synonym_rep(text, n=30)
    elif choice == "typo":
        text = random_typo(text, prob=0.4)
    elif choice == "swap":
        text = random_swap(text, n=10)
    elif choice == "del":
        text = random_deletion(text, p=0.2)
    elif choice == "add":
        text = random_insert(text, n=10)
    
    example["text"] = text
    return example
    
    # You should update example["text"] using your transformation

    
    
    # raise NotImplementedError

    ##### YOUR CODE ENDS HERE ######

    return example
