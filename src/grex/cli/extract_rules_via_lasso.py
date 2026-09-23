import argparse
import json

from grex.data.data import CorpusData, ExtractionConfig
from grex.extract.lasso import LassoExtractor
from grex.utils.encoding import NumpyEncoder

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("data", metavar="F", type=str, nargs="+")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--patterns", required=True)
    cmd.add_argument("--config", default="ud")
    cmd.add_argument("--max-degree", type=int, default=1)
    cmd.add_argument("--min-feature_occurence", type=int, default=5)
    cmd.add_argument("--alpha-start", type=float, default=0.1)
    cmd.add_argument("--alpha-end", type=float, default=0.001)
    cmd.add_argument("--alpha-num", type=int, default=100)
    args = cmd.parse_args()

    config = ExtractionConfig.from_yaml(args.patterns)
    corpus = CorpusData.from_corpus(args.data, config, args.config)
    m = corpus.to_matrix(args.max_degree, args.min_feature_occurence)

    extractor = LassoExtractor(
        config.scope,
        config.conclusion,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        alpha_num=args.alpha_num,
    )

    X = m.X
    y = m.binarize(config.positive)

    extractor.fit(X, y, m.feature_names)

    print("Done.", flush=True)
    with open(args.output, "w") as out:
        json.dump(extractor.to_dict(), out, cls=NumpyEncoder, indent=3)
