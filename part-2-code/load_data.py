import os, random, re, string, difflib
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import sqlite3
import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
PAD_IDX = tokenizer.pad_token_id





class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.data_folder = data_folder
        self.split = split
        self.tokenizer = tokenizer
        self.inputs, self.targets = self.process_data(data_folder, split, self.tokenizer)
        

    def process_data(self, data_folder, split, tokenizer):
        
        nl_path = os.path.join(data_folder, f"{split}.nl")
        inputs = load_lines(nl_path)
        
        if split in ["train", "dev"]:
            sql_path = os.path.join(data_folder, f"{split}.sql")
            targets = load_lines(sql_path)
        else:
            targets = [""] * len(inputs)
        
        assert len(inputs) == len(targets), f"Length mismath in {split} data."
        
        return inputs, targets
        
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        
        input_text = f"translate English to SQL: {self.inputs[idx].strip()}"        

        target_text = normalize_sql(self.targets[idx])
        #target_text = self.targets[idx].strip()
    
        enc = self.tokenizer.encode(
            input_text,
            truncation=True,
            max_length=512,
            add_special_tokens=True,       
        )
        if len(enc) == 0:
            enc = [self.tokenizer.pad_token_id]  
    
        if self.split != "test":
            dec = self.tokenizer.encode(
                target_text,
                truncation=True,
                max_length=512,
                add_special_tokens=True,    
            )

    
            return torch.tensor(enc, dtype=torch.long), torch.tensor(dec, dtype=torch.long)
        else:
            return torch.tensor(enc, dtype=torch.long)


def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''

    encoder_seqs = []
    decoder_seqs = []

    for enc_ids, dec_ids in batch:
        if enc_ids.numel() == 0:
            enc_ids = torch.tensor([PAD_IDX], dtype=torch.long)
        if dec_ids.numel() == 0:
            dec_ids = torch.tensor([PAD_IDX], dtype=torch.long)
        encoder_seqs.append(enc_ids)
        decoder_seqs.append(dec_ids)

    encoder_idx = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    decoder_idx = pad_sequence(decoder_seqs, batch_first=True, padding_value=PAD_IDX)

    encoder_mask = encoder_idx.ne(PAD_IDX) 
    

    decoder_inputs = decoder_idx[:, :-1] 
    decoder_targets = decoder_idx         


    bos = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    initial_decoder_inputs = torch.full((encoder_idx.size(0), 1), bos, dtype=torch.long)


    return (
        encoder_idx.contiguous(),
        encoder_mask.contiguous(),
        decoder_inputs.contiguous(),     
        decoder_targets.contiguous(),    
        initial_decoder_inputs.contiguous(),
    )


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_seqs = []
    for enc_ids in batch:
        if enc_ids.numel() == 0:
            enc_ids = torch.tensor([PAD_IDX], dtype=torch.long)
        encoder_seqs.append(enc_ids)

    encoder_idx = pad_sequence(encoder_seqs, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = encoder_idx.ne(PAD_IDX)

    bos = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    initial_decoder_inputs = torch.full((len(batch), 1), bos, dtype=torch.long)

    return (
    encoder_idx.contiguous(),
    encoder_mask.contiguous(),
    initial_decoder_inputs.contiguous(),
)




def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x


def normalize_sql(sql):
    sql = sql.strip()
    sql = sql.lower()


    # Normalize parentheses spacing
    sql = sql.replace("(", " ( ").replace(")", " ) ")

    # Normalize quotes
    sql = sql.replace('"', "'")

    sql = re.sub(r"\s+", " ", sql)

    # Remove trailing semicolons
    sql = sql.rstrip(";")

    return sql.strip()







