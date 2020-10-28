import os
import random
from copy import deepcopy
import logging
import time
import json

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor as sp

from config import Config


class Reader:
    def __init__(self, config):
        self.tokenizer = sp(config.kogpt2_tokenizer_path)
        self.train_data = []
        self.dev_data = []
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.max_length = config.max_length
        self.vocab_size = config.vocab_size
        self.bos_idx = config.bos_idx
        self.eos_idx = config.eos_idx
        self.pad_idx = config.pad_idx

    def load_data(self):
        self.train_data = open(os.path.join(self.data_path, "train_data.txt"), "r").read().split("\n")[:-1]
        self.dev_data = open(os.path.join(self.data_path, "dev_data.txt"), "r").read().split("\n")[:-1]

    def make_batch(self, mode="train"):
        if mode == "train":
            data = self.train_data
        else:
            data = self.dev_data
        all_batches = []
        batch = []
        for row in data:
            batch.append(row)
            if len(batch) == self.batch_size:
                all_batches.append(batch)
                batch = []
        if len(batch) > 0:
            all_batches.append(batch)
        random.shuffle(all_batches)
        for batch in all_batches:
            yield batch
                
    def make_input(self, batch):
        batch_size = len(batch)
        inputs = torch.ones(batch_size, self.max_length, dtype=torch.int64).cuda() * self.pad_idx
        labels = torch.ones(batch_size, self.max_length, dtype=torch.int64).cuda() * self.pad_idx
        max_length = 0
        for batch_idx in range(batch_size):
            tokens = self.tokenizer.EncodeAsIds(batch[batch_idx])
            length = len(tokens)
            inputs[batch_idx, :length] = torch.tensor(tokens, dtype=torch.int64)
            labels[batch_idx, :length] = torch.tensor(tokens[1:] + [self.eos_idx], dtype=torch.int64)
            max_length = max(max_length, length)
        inputs = inputs[:, :max_length]
        labels = labels[:, :max_length]

        return inputs, labels


if __name__ == "__main__":
    config = Config()
    parser = config.parser
    config = parser.parse_args()
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)

    reader = Reader(config)
    logger.info("Load data...")
    start = time.time()
    reader.load_data()
    end = time.time()
    logger.info("{} secs".format(end-start))

    logger.info("Make batch...")
    start = time.time()
    iterator = reader.make_batch("dev")
    end = time.time()
    logger.info("{} secs".format(end-start))

    for batch in iterator:
        inputs = reader.make_input(batch)

