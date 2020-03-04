# -*- coding: utf-8 -*-

import yaml
import argparse

from model import InstaGAN
from trainer import Trainer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("param_file")

    args = parser.parse_args()

    with open(args.param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = Trainer(params)

    trainer.train()