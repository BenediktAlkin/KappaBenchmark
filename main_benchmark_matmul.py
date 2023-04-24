import os
import torch
from argparse import ArgumentParser
from tqdm import tqdm
import logging
import sys
import kappaprofiler as kp

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dim", type=int, default=65536)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=int, default=0)
    return vars(parser.parse_args())

def work(x):
    x @ x

def main(dim, epochs, device):
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=f"%(asctime)s %(message)s", datefmt="%m-%d %H:%M:%S")
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    logger.handlers.append(handler)

    # setup GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup matrix
    x = torch.randn(dim, dim, device=device)

    warmup_epochs = epochs // 10
    logging.info(f"warmup for {warmup_epochs} epochs")
    with kp.Stopwatch() as sw:
        for _ in tqdm(range(warmup_epochs)):
            work(x)
    logging.info(f"warmup took {sw.elapsed_seconds} seconds")

    logging.info(f"start working")
    with kp.Stopwatch() as sw:
        for _ in tqdm(range(epochs)):
            work(x)
    logging.info(f"work took {sw.elapsed_seconds} seconds")



if __name__ == "__main__":
    main(**parse_args())