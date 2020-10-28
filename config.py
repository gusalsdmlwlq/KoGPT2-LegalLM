import argparse


class Config:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="data/gpt", type=str)
    parser.add_argument("--kogpt2_model_path", default="save/kogpt2.pt", type=str)
    parser.add_argument("--kogpt2_tokenizer_path", default="save/kogpt2.sp", type=str)
    parser.add_argument("--bos_idx", default=0, type=int)
    parser.add_argument("--eos_idx", default=1, type=int)
    parser.add_argument("--pad_idx", default=3, type=int)
    parser.add_argument("--unk_idx", default=5, type=int)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--vocab_size", default=50000, type=int)
    parser.add_argument("--batch_size", default="2", type=int)
    parser.add_argument("--num_gpus", default="1", type=int)
    parser.add_argument("--lr", default="3e-5", type=float)
    parser.add_argument("--seed", default="1", type=int)
    parser.add_argument("--save_path", default=None, type=str)
    parser.add_argument("--early_stop_count", default="5", type=int)
    parser.add_argument("--max_epochs", default="40", type=int)
    parser.add_argument("--min_epochs", default="20", type=int)
    parser.add_argument("--dynamic_tqdm", action="store_true", default=False)
    parser.add_argument("--master_port", default=29500, type=int)
