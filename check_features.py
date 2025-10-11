import argparse
import yaml

from grex.data import extract_data
from grex.utils import FeaturePredicate

cmd = argparse.ArgumentParser()
cmd.add_argument('data', metavar='F', type=str, nargs='+', help='data')
cmd.add_argument("--patterns", type=str, required=True)
cmd.add_argument("--config", type=str, default="ud")
args = cmd.parse_args()

with open(args.patterns) as instream:
    config = yaml.load(instream, Loader=yaml.Loader)

scope = config["scope"]
conclusion = config.get("conclusion", None)

templates = FeaturePredicate.from_config(config["templates"])
feature_predicate = FeaturePredicate.from_config(config["features"], templates=templates)

data = extract_data(args.data, scope, conclusion, feature_predicate, config=args.config)

available_features = set()
for sentence in data:
    available_features.update(sentence["input"].keys())

for f in sorted(available_features):
    print(f)
