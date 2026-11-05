import pandas as pd
from src.paraphrasepipeline import ParaphrasePipeline
from src.trainconfig import parse_config
from src.utils import load_datasets, make_encoder, make_model


def train_model(pipeline: ParaphrasePipeline, train_df: pd.DataFrame, dev_df: pd.DataFrame) -> None:
    pipeline.fit(train_df)

    dev_pred = pipeline.predict(dev_df)

    print((dev_pred == dev_df["label"]).mean())


def main() -> None:
    config = parse_config()
    train_df, dev_df, test_df = load_datasets()

    train_df = train_df.sample(10)
    dev_df = dev_df.sample(10)
    pipeline = ParaphrasePipeline(make_encoder(config), make_model(config))

    train_model(pipeline, train_df, dev_df)


if __name__ == "__main__":
    main()
