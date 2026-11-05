from argparse import ArgumentParser
from dataclasses import dataclass

ENCODER_TO_SOLVER = {
    "indicator": "saga",
    "bow": "saga",
    "sentencetransformer": "lbfgs",
}

@dataclass
class TrainConfig:
    encoder: str
    model: str

    @property
    def loglinear_solver(self) -> str:
        return ENCODER_TO_SOLVER[self.encoder]


def parse_config() -> TrainConfig:
    parser = ArgumentParser()
    parser.add_argument("--encoder", choices=["indicator", "bow", "sentencetransformer"], required=True)
    parser.add_argument("--model", choices=["loglinear", "shallowmlp", "deepmlp"], required=True)
    return TrainConfig(**vars(parser.parse_args()))
