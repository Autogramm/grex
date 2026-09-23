import argparse

from grex.data.data import CorpusData, ExtractionConfig

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("data", metavar="F", type=str, nargs="+")
    cmd.add_argument("--patterns", required=True)
    cmd.add_argument("--config", default="ud")
    args = cmd.parse_args()

    config = ExtractionConfig.from_yaml(args.patterns)
    corpus = CorpusData.from_corpus(args.data, config, args.config)

    for f in corpus.unique_features():
        print(f)
