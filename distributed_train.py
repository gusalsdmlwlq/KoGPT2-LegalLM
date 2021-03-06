import os
import logging
import time
import random
import math
import re
from copy import deepcopy

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import Process
import torch.nn.functional as F
from tqdm import tqdm
import torch.distributed as dist
from sentencepiece import SentencePieceProcessor as sp
from transformers import GPT2Config, GPT2LMHeadModel
from nltk import bleu_score

from config import Config
from reader import Reader


def distribute_data(batches, num_gpus):
    distributed_data = []
    if len(batches) % num_gpus == 0:
        batch_size = int(len(batches) / num_gpus)
        for idx in range(num_gpus):
            distributed_data.append(batches[batch_size*idx:batch_size*(idx+1)])
    else:
        batch_size = math.ceil(len(batches) / num_gpus)
        expanded_batches = deepcopy(batches) if type(batches) == list else batches.clone()
        while True:
            expanded_batches = expanded_batches + deepcopy(batches) if type(batches) == list else torch.cat([expanded_batches, batches.clone()], dim=0)
            if len(expanded_batches) >= batch_size*num_gpus:
                expanded_batches = expanded_batches[:batch_size*num_gpus]
                break
        for idx in range(num_gpus):
            distributed_data.append(expanded_batches[batch_size*idx:batch_size*(idx+1)])

    return distributed_data

def init_process(local_rank, backend, config):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = "{}".format(config.master_port)
    dist.init_process_group(backend, rank=local_rank, world_size=config.num_gpus)
    torch.cuda.set_device(local_rank)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    if local_rank != 0:
        logger.setLevel(logging.WARNING)
    
    if local_rank == 0:
        writer = SummaryWriter()
        if not os.path.exists("save"):
            os.mkdir("save")
        save_path = "save/model_{}.pt".format(re.sub("\s+", "_", time.asctime()))

    reader = Reader(config)
    start = time.time()
    logger.info("Loading data...")
    reader.load_data()
    end = time.time()
    logger.info("Loaded. {} secs".format(end-start))

    gpt2_config = GPT2Config(vocab_size=config.vocab_size)
    model = GPT2LMHeadModel(gpt2_config).cuda()
    model.load_state_dict(torch.load(config.kogpt2_model_path, map_location = lambda storage, loc: storage.cuda(local_rank)))
    optimizer = Adam(model.parameters(), lr=config.lr)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    train.global_step = 0
    train.max_iter = len(list(reader.make_batch("train")))
    validate.max_iter = len(list(reader.make_batch("dev")))
    global_epoch = 0

    min_loss = 1e+9
    lr = config.lr
    early_stop_count = config.early_stop_count

    if config.save_path is not None:
        step, epoch, loss = load(model, optimizer, config.save_path, local_rank, config)
        train.global_step = step
        global_epoch = epoch
        min_loss = loss

    logger.info("Validate...")
    loss = validate(model, reader, config, local_rank)
    logger.info("Loss: {:.4f}".format(loss))

    model.train()

    for epoch in range(global_epoch, global_epoch + config.max_epochs):
        logger.info("Train...")
        start = time.time()

        if local_rank == 0:
            train(model, reader, optimizer, config, local_rank, epoch, min_loss, writer)
        else:
            train(model, reader, optimizer, config, local_rank, epoch, min_loss)
        
        end = time.time()
        global_epoch = epoch
        logger.info("epoch: {}, {:.4f} secs".format(epoch+1, end-start))

        logger.info("Validate...")
        loss = validate(model, reader, config, local_rank)
        logger.info("Loss: {:.4f}".format(loss))
        
        if local_rank == 0:
            writer.add_scalar("Val/Loss", loss, epoch+1)

        if loss < min_loss:  # save model
            if local_rank == 0:
                save(model, optimizer, save_path, train.global_step, epoch, loss)
                logger.info("Saved to {}.".format(os.path.abspath(save_path)))
            
            min_loss = loss
            early_stop_count = config.early_stop_count
        else:  # ealry stopping
            if early_stop_count == 0:
                if epoch < config.min_epochs:
                    early_stop_count += 1
                    logger.info("Too early to stop training.")
                    logger.info("early stop count: {}".format(early_stop_count))
                else:
                    logger.info("Early stopped.")
                    break
            elif early_stop_count == 2:
                lr = lr / 2
                logger.info("learning rate schedule: {}".format(lr))
                for param in optimizer.param_groups:
                    param["lr"] = lr
            early_stop_count -= 1
            logger.info("early stop count: {}".format(early_stop_count))
    logger.info("Training finished.")

def train(model, reader, optimizer, config, local_rank, global_epoch, min_loss, writer=None):
    iterator = reader.make_batch("train")

    if local_rank == 0:  # only one process prints something
        t = tqdm(enumerate(iterator), total=train.max_iter, ncols=150, position=0, leave=True, dynamic_ncols=config.dynamic_tqdm)
    else:
        t = enumerate(iterator)

    for batch_idx, batch in t:
        try:
            inputs, labels = reader.make_input(batch)
            batch_size = inputs.size(0)
            length = inputs.size(1)
            distributed_batch_size = math.ceil(batch_size / config.num_gpus)

            # distribute batches to each gpu
            inputs = distribute_data(inputs, config.num_gpus)[local_rank].cuda().contiguous()
            labels = distribute_data(labels, config.num_gpus)[local_rank].cuda().contiguous()

            model.zero_grad()
            pad_mask = (inputs != reader.pad_idx).cuda()
            labels.masked_fill_(~pad_mask, value=-100)
            pred = model(inputs, attention_mask=pad_mask)[0]
            loss = F.cross_entropy(pred.view(-1, config.vocab_size), labels.contiguous().view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            train.global_step += 1

            if local_rank == 0:
                writer.add_scalar("Train/Loss", loss.item(), train.global_step)
                t.set_description("iter: {}, loss: {:.4f}".format(batch_idx+1, loss.item()))
                time.sleep(1)
            
            del pred, loss
            torch.cuda.empty_cache()

        except RuntimeError as e:
            print(e)
            print("batch size: {}, length: {}".format(batch_size, length))
            error_save_path = "save/model_error_{}.pt".format(re.sub("\s+", "_", time.asctime()))
            print("model saved to {}".format(error_save_path))
            save(model, optimizer, error_save_path, train.global_step, global_epoch, min_loss)
            exit(0)

        except KeyboardInterrupt as e:
            print(e)
            stop_save_path = "save/model_stop_{}.pt".format(re.sub("\s+", "_", time.asctime()))
            print("model saved to {}".format(stop_save_path))
            save(model, optimizer, stop_save_path, train.global_step, global_epoch, min_loss)
            exit(0)

# def validate(model, reader, config, local_rank):
#     model.eval()
#     bleu = 0
#     batch_count = 0
#     bleu_smoothing = bleu_score.SmoothingFunction().method7
#     with torch.no_grad():
#         iterator = reader.make_batch("dev")

#         if local_rank == 0:
#             t = tqdm(enumerate(iterator), total=validate.max_iter, ncols=150, position=0, leave=True)
#         else:
#             t = enumerate(iterator)

#         for batch_idx, batch in t:
#             inputs, labels = reader.make_input(batch)
#             batch_size = inputs.size(0)
#             length = inputs.size(1)
#             words = []
#             eos_batches = [False for i in range(batch_size)]
#             end_batches = [True for i in range(batch_size)]
#             for word_count in range(config.max_length):
#                 pad_mask = (inputs != reader.pad_idx).cuda()
#                 outputs = model(inputs, attention_mask=pad_mask)
#                 pred = outputs[0][:, -1, :].detach()
#                 word = pred.argmax(dim=-1)
#                 words.append(word)
#                 word = word.tolist()
#                 new_inputs = torch.ones(batch_size, min(inputs.size(1)+1, config.max_length), dtype=torch.int64).cuda() * config.pad_idx
#                 for b_idx in range(batch_size):
#                     if eos_batches[b_idx]:
#                         continue
#                     b_input = inputs[b_idx][inputs[b_idx] != reader.pad_idx].tolist()
#                     b_input.append(word[b_idx])
#                     b_input = b_input[-config.max_length:]
#                     if word[b_idx] == reader.eos_idx:
#                         eos_batches[b_idx] = True
#                     new_inputs[b_idx, :len(b_input)] = torch.tensor(b_input, dtype=torch.int64)
#                 inputs = new_inputs
#                 del new_inputs, outputs
#                 if eos_batches == end_batches:
#                     break
#             words = torch.stack(words, dim=1).tolist()
#             for b_idx in range(batch_size):
#                 label = labels[b_idx]
#                 label[label != config.pad_idx].tolist()
#                 bleu += bleu_score.sentence_bleu([label[label != config.pad_idx].tolist()], words[b_idx], smoothing_function=bleu_smoothing)
#             batch_count += batch_size

#             if local_rank == 0:
#                 t.set_description("iter: {}".format(batch_idx+1))
#                 time.sleep(1)
#             torch.cuda.empty_cache()
#     model.train()
#     bleu = bleu / batch_count

#     return bleu

def validate(model, reader, config, local_rank):
    model.eval()
    loss = 0
    batch_count = 0
    with torch.no_grad():
        iterator = reader.make_batch("dev")

        if local_rank == 0:
            t = tqdm(enumerate(iterator), total=validate.max_iter, ncols=150, position=0, leave=True)
        else:
            t = enumerate(iterator)

        for batch_idx, batch in t:
            inputs, labels = reader.make_input(batch)
            batch_size = inputs.size(0)

            model.zero_grad()
            pad_mask = (inputs != reader.pad_idx).cuda()
            labels.masked_fill_(~pad_mask, value=-100)
            pred = model(inputs, attention_mask=pad_mask)[0]
            loss += F.cross_entropy(pred.view(-1, config.vocab_size), labels.contiguous().view(-1), ignore_index=-100).item() * batch_size
            batch_count += batch_size

            if local_rank == 0:
                t.set_description("iter: {}".format(batch_idx+1))
                time.sleep(1)
            torch.cuda.empty_cache()
    model.train()
    loss = loss / batch_count

    return loss

def save(model, optimizer, save_path, step, epoch, loss):
    checkpoint = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "loss": loss
    }
    torch.save(checkpoint, save_path)

def load(model, optimizer, save_path, local_rank, config):
    checkpoint = torch.load(save_path, map_location = lambda storage, loc: storage.cuda(local_rank))
    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["step"], checkpoint["epoch"], checkpoint["loss"]

if __name__ == "__main__":
    os.environ["KMP_WARNINGS"] = "0"
    config = Config()
    parser = config.parser
    config = parser.parse_args()
    processes = []
    try:
        torch.multiprocessing.set_start_method('spawn')
        for local_rank in range(config.num_gpus):
            process = Process(target=init_process, args=(local_rank, config.backend, config))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
    except RuntimeError:
        ctx = torch.multiprocessing.get_context("spawn")
        for local_rank in range(config.num_gpus):
            process = ctx.Process(target=init_process, args=(local_rank, config.backend, config))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()