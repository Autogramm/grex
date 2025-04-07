import argparse
import yaml
import json
import numpy as np
import skglm
import scipy.stats

from grex.data import extract_data
from grex.utils import FeaturePredicate

import pyximport
pyximport.install()
import grex.features

if __name__ == "__main__":
    cmd = argparse.ArgumentParser()
    cmd.add_argument('data', metavar='F', type=str, nargs='+', help='data')
    cmd.add_argument("--output", type=str, required=True)
    cmd.add_argument("--patterns", type=str, required=True)
    cmd.add_argument("--config", type=str, default="ud")
    cmd.add_argument("--max-degree", type=int, default=1)
    cmd.add_argument("--min-feature_occurence", type=int, default=5)
    cmd.add_argument("--alpha-start", type=float, default=0.1)
    cmd.add_argument("--alpha-end", type=float, default=0.001)
    cmd.add_argument("--alpha-num", type=int, default=100)
    args = cmd.parse_args()

    with open(args.patterns) as instream:
        config = yaml.load(instream, Loader=yaml.Loader)

    scope = config["scope"]
    conclusion = config.get("conclusion", None)
    conclusion_meta = config.get("conclusion_meta", None)

    templates = FeaturePredicate.from_config(config["templates"])
    feature_predicate = FeaturePredicate.from_config(config["features"], templates=templates)

    print("Loading dataset...", flush=True)
    include_metadata = any('meta' in k for k in config.get('features', {}).get('sentence', {}))
    data = extract_data(args.data, scope, conclusion, conclusion_meta, feature_predicate, args.config, include_metadata)

    # quick checks
    if len(data) == 0:
        raise RuntimeError("Patterns resulted in empty dataset")
    num_positive = sum(sentence["output"] for sentence in data)
    if num_positive == 0:
        raise RuntimeError("The conclusion does not appear in the dataset")
    if num_positive == len(data):
        raise RuntimeError("The conclusion always appears in the dataset")

    data_inputs = list()
    data_outputs = list()
    for sentence in data:
        data_inputs.append(sentence["input"])
        data_outputs.append(sentence["output"])

    print("Number of occurences of the conclusion: %i / %i" % (num_positive, len(data)))

    print("Extracting features", flush=True)
    feature_set = grex.features.FeatureSet()

    feature_set.add_feature(grex.features.AllSingletonFeatures())
    for degree in range(2, args.max_degree + 1):
        feature_set.add_feature(grex.features.AllProductFeatures(
            degree=degree,
            min_occurences=args.min_feature_occurence
        ))

    try:
        feature_set.init_from_data(data_inputs)
        X = feature_set.build_features(data_inputs, sparse=True)
        if X.shape[1] == 0:
            raise RuntimeError("Empty feature list!")
    except RuntimeError:
        RuntimeError("There was an error during feature extraction")

    # build targets
    y = np.empty((len(data),), dtype=np.int_)
    for i, v in enumerate(data_outputs):
        assert v in [0, 1]
        y[i] = v

    extracted_rules = dict()
    extracted_rules['scope'] = scope
    if conclusion_meta:
        meta = ",".join(f"{k}={v}" for k, v in conclusion_meta.items())
        extracted_rules['conclusion'] = f"{conclusion},{meta}" if conclusion else meta
    else:
        extracted_rules['conclusion'] = conclusion or ""
    extracted_rules["data_len"] = len(data)
    extracted_rules["n_yes"] = num_positive
    extracted_rules["intercepts"] = list()

    # extract rules
    all_rules = set()
    ordered_rules = list()

    alphas = alphas=np.linspace(args.alpha_start, args.alpha_end, args.alpha_num)
    for j, alpha in enumerate(alphas):
        print("extracting rules (%i / %i)" % (j + 1, len(alphas)), flush=True)
        model = skglm.SparseLogisticRegression(
            alpha=alpha,
            fit_intercept=True,
            max_iter=20,
            max_epochs=1000,
        )
        model.fit(X, y)
        extracted_rules["intercepts"].append((alpha, model.intercept_))

        for name, (value, idx) in feature_set.feature_weights(model.coef_[0]).items():
            if name not in all_rules:
                all_rules.add(name)
                col = np.asarray(X[:, idx].todense())
                idx_col = col.squeeze(1)

                with_feature_selector = idx_col > 0
                without_feature_selector = np.logical_not(with_feature_selector)

                matched = y[with_feature_selector]
                n_matched = len(matched)
                n_pattern_positive_occurence = matched.sum()
                n_pattern_negative_occurence = n_matched - n_pattern_positive_occurence

                mu = (num_positive / len(data))
                a = (n_pattern_positive_occurence / n_matched)
                gstat = 2 * n_matched * (
                        ((a * np.log(a)) if a > 0 else 0) - a * np.log(mu)
                        + (((1 - a) * np.log(1 - a)) if (1 - a) > 0 else 0) - (1 - a) * np.log(1 - mu)
                )
                p_value = 1 - scipy.stats.chi2.cdf(gstat, 1)
                cramers_phi = np.sqrt((gstat / n_matched))

                expected = (n_matched * num_positive) / len(data)
                delta_observed_expected = n_pattern_positive_occurence - expected

                coverage_q = n_pattern_positive_occurence / n_matched
                coverage_not_q = n_pattern_negative_occurence / n_matched

                if n_pattern_positive_occurence / n_matched > int(y.sum()) / len(data):
                    decision = 'yes'
                    coverage = (n_pattern_positive_occurence / num_positive) * 100
                    precision = (n_pattern_positive_occurence / n_matched) * 100
                    ratio = coverage_q / coverage_not_q if coverage_not_q != 0 else 0
                else:
                    decision = 'no'
                    coverage = (n_pattern_negative_occurence / (len(data) - num_positive)) * 100
                    precision = (n_pattern_negative_occurence / n_matched) * 100
                    ratio = coverage_not_q / coverage_q if coverage_q != 0  else 0

                ordered_rules.append({
                    "pattern": name,
                    "n_pattern_occurences": int(idx_col.sum()),
                    "n_pattern_positive_occurences": int(n_pattern_positive_occurence),
                    "n_pattern_negative_occurrences": int(n_pattern_negative_occurence),
                    "decision": decision,
                    "alpha": alpha,
                    "value": value,
                    "coverage": coverage,
                    "coverage_q_in_p": coverage_q,
                    "coverage_not_q_in_p": coverage_not_q,
                    "precision": precision,
                    "ratio": ratio,
                    "delta": delta_observed_expected,
                    "g-statistic": gstat,
                    "p-value": p_value,
                    "cramers_phi": cramers_phi
                })

    extracted_rules["rules"] = ordered_rules

    # if len(extracted_data) == 3:
    #    break

print("Done.", flush=True)
with open(args.output, 'w') as out_stream:
    json.dump(extracted_rules, out_stream)
