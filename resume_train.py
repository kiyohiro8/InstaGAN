
import yaml
import argparse

from model import InstaGAN
from trainer import Trainer

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("resume_from")

    args = parser.parse_args()

    param_file = f"{args.resume_from}/params.json" 

    with open(param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    trainer = Trainer(params)

    trainer.train(resume_from=args.resume_from)