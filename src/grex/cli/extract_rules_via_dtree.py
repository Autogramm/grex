import argparse
import json

from grex.data.data import CorpusData, ExtractionConfig
from grex.extract.dtree import DTreeExtractor
from grex.utils.encoding import NumpyEncoder

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument("data", metavar="F", type=str, nargs="+")
    cmd.add_argument("--output", required=True)
    cmd.add_argument("--patterns", required=True)
    cmd.add_argument("--config", default="ud")
    cmd.add_argument("--grew", action="store_true")
    cmd.add_argument("--min-samples_leaf", type=int, default=5)
    cmd.add_argument(
        "--node_impurity", type=float, default=0.15
    )
    cmd.add_argument("--threshold", type=float, default=1e-1)
    cmd.add_argument("--max-depth", type=int, default=12)
    cmd.add_argument("--only-leaves", action="store_true")
    args = cmd.parse_args()

    config = ExtractionConfig.from_yaml(args.patterns)
    corpus = CorpusData.from_corpus(args.data, config, args.config)
    m = corpus.to_matrix(max_degree=1, min_feature_occurence=1)

    extractor = DTreeExtractor(
        config.scope,
        config.conclusion,
        min_samples_leaf=args.min_samples_leaf,
        max_depth=args.max_depth,
        threshold=args.threshold,
        only_leaves=args.only_leaves,
        grew=args.grew,
    )

    extractor.fit(m.X, m.binarize(config.positive), m.feature_names)

    print("Done.", flush=True)
    with open(args.output, "w") as out:
        json.dump(extractor.to_dict(), out, indent=3, cls=NumpyEncoder)
