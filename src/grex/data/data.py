from dataclasses import dataclass

import grewpy
import numpy as np
import yaml
from scipy.sparse import spmatrix
from sklearn.preprocessing import LabelEncoder

from grex.data.features import AllProductFeatures, AllSingletonFeatures, FeatureSet
from grex.data.predicate import FeaturePredicate

# le terme "similar" n'est pas du tout adapté,
# mais c'est en gros les POS tags qu'on veut merger ensemble
# pour ajouter de nouvelles features
# càd que plutôt qu'avoir seulement "POS=NOUN",
# on veut aussi avoir "POS=NOUN|PROPN"
SIMILAR_POS_TAGS = [
    # use list to fix order, probably unecessary,
    # but just in case
    ["NOUN", "PROPN"],
    ["NOUN", "PROPN", "PRON"],
    ["AUX", "VERB"],
    ["DET", "NUM"],
]


def parents_from_successors(sucs):
    parents = dict()
    for head, children in sucs.items():
        for cid, rel in children:
            assert cid not in parents
            parents[cid] = head, rel
    return parents


def build_node_features(node_name, relation_name, features, predicate):
    ret = dict()
    for k, v in features.items():
        if predicate(node_name, relation_name, k, v):
            if k == "lemma":
                # for lemmas we need to keep the upos information for filtering
                ret[("node", node_name, relation_name, k)] = (v, features["upos"])
            else:
                ret[("node", node_name, relation_name, k)] = v
    return ret


def extract_features(draft, match, feature_predicate):
    sentence = draft[match["sent_id"]]
    parents = parents_from_successors(sentence.sucs)
    selected_node_ids = set(match["matching"]["nodes"].values())
    features = dict()

    # sentence level meta features
    for k, v in sentence.meta.items():
        try:
            if feature_predicate("sentence", "meta", k):
                features[("sentence", "meta", k)] = v
        except KeyError:
            break

    for node_name, node_id in match["matching"]["nodes"].items():
        if node_id == "0":  # fake root
            continue

        # node features
        features.update(
            build_node_features(node_name, "own", sentence.features[node_id], feature_predicate)
        )
        # parent node
        parent_id, rel = parents[node_id]
        rel_v = rel.get("deep") or ":".join(rel[k] for k in ("1", "2") if k in rel)

        # node relation
        if feature_predicate(node_name, "own", "rel_shallow", rel_v):
            k = ("node", node_name, "own", "rel_shallow")
            features[("node", node_name, "own", "rel_shallow")] = rel_v

        if feature_predicate(node_name, "own", "rel_deep") and "deep" in rel:
            k = ("node", node_name, "own", "rel_deep")
            features[("node", node_name, "own", "rel_deep")] = rel_v

        # position of parent
        if parent_id != "0":
            position_v = "before" if int(parent_id) < int(node_id) else "after"
            if feature_predicate(node_name, "parent", "position", position_v):
                features[("node", node_name, "parent", "position")] = position_v

        # parent features
        if parent_id not in selected_node_ids:
            features.update(
                build_node_features(
                    node_name, "parent", sentence.features[parent_id], feature_predicate
                )
            )

        # prev word
        prev_id = str(int(node_id) - 1)
        if prev_id in sentence.features and prev_id not in selected_node_ids:
            features.update(
                build_node_features(
                    node_name, "prev", sentence.features[prev_id], feature_predicate
                )
            )

        # next word
        next_id = str(int(node_id) + 1)
        if next_id in sentence.features and next_id not in selected_node_ids:
            features.update(
                build_node_features(
                    node_name, "next", sentence.features[next_id], feature_predicate
                )
            )

        # children features
        children = sentence.sucs.get(node_id, list())
        if not children and feature_predicate(node_name, "own", "has_child"):
            features[("node", node_name, "own", "has_child")] = "No"

        for child_id, child_rel in children:
            # only consider children that are not matched nodes
            if child_id in selected_node_ids:
                continue

            # child features
            child_features = sentence.features[child_id]
            for k, v in child_features.items():
                if feature_predicate(node_name, "child", k, v):
                    key = ("node", node_name, "child", k)
                    if k == "lemma":
                        # for lemmas we need to keep the upos information for filtering
                        if key not in features:
                            features[key] = {(v, child_features["upos"])}
                        else:
                            features[key].add((v, child_features["upos"]))
                    else:
                        if key not in features:
                            features[key] = {v}
                        else:
                            features[key].add(v)

            # relation with child
            _, rel = parents[child_id]
            child_rel_v = rel.get("deep") or ":".join(rel[k] for k in ("1", "2") if k in rel)
            if feature_predicate(node_name, "child", "rel_shallow", child_rel_v):
                k = ("node", node_name, "child", "rel_shallow")
                v = ":".join(rel[rel_key] for rel_key in ["1", "2"] if rel_key in rel)
                if k not in features:
                    features[k] = {v}
                else:
                    features[k].add(v)
            if feature_predicate(node_name, "child", "rel_deep", child_rel_v) and "deep" in rel:
                k = ("node", node_name, "child", "rel_deep")
                v = rel["deep"]
                if k not in features:
                    features[k] = {v}
                else:
                    features[k].add(v)
    return features


def extract_data(treebank_paths, scope, conclusion, feature_predicate, config="ud"):
    grewpy.set_config(config)

    if conclusion is None:
        raise RuntimeError("No conclusion provided in configuration")

    if isinstance(treebank_paths, str):
        treebank_paths = [treebank_paths]

    data = []
    for tp in treebank_paths:
        corpus = grewpy.Corpus(tp)
        draft = grewpy.CorpusDraft(corpus)

        req = grewpy.Request(scope)
        if conclusion is not None:
            # TODO: check conclusion string, probably line 198 and 199 should be change too
            matches = corpus.search(req, clustering_keys=[f"{{ {conclusion} }}"])
            if matches == []:
                raise RuntimeError("There are no matches for the specified scope.")
            matches = [(sent, c, sent["sent_id"]) for c, sents in matches.items() for sent in sents]
        else:
            matches = [(sent, "Yes", sent["sent_id"]) for sent in corpus.search(req)]

        for match, c, sent_id in matches:
            # TODO: add a test

            data.append(
                {
                    "input": extract_features(draft, match, feature_predicate),
                    "sent_id": sent_id,
                    "output": c,  # TODO 1 if c == "Yes" else 0
                }
            )

        corpus.clean()
    # now we filter lemmas
    # we first need to find allowed lemmas
    feature_predicate.reset_lemmas_counter()
    for match in data:
        for k, feature_value in match["input"].items():
            # only node features have lemmas
            if k[0] != "node" or k[-1] != "lemma":
                continue

            # at this point we are checking a lemma feature
            _, node_name, relation_name, feature_name = k

            if type(feature_value) is set:
                for lemma, upos in feature_value:
                    feature_predicate.update_lemmas_counter(node_name, relation_name, lemma, upos)
            else:
                feature_predicate.update_lemmas_counter(
                    node_name, relation_name, feature_value[0], feature_value[1]
                )
    feature_predicate.freeze_lemmas_counter()

    # transform keys to strings
    # and filter lemmas.
    sanitized_data = list()
    for match in data:
        sanitized_input = dict()
        for k, feature_value in match["input"].items():
            is_lemma = k[0] == "node" and k[-1] == "lemma"
            if is_lemma:
                if type(feature_value) is set:
                    feature_value = {
                        lemma
                        for lemma, upos in feature_value
                        if feature_predicate.check_lemma(k[1], k[2], lemma, upos)
                    }
                    # skip if no len anymore
                    if len(feature_value) == 0:
                        continue
                else:
                    # skip if not allowed
                    if not feature_predicate.check_lemma(
                        k[1], k[2], feature_value[0], feature_value[1]
                    ):
                        continue
                    feature_value = feature_value[0]
            sanitized_input[":".join(k)] = feature_value

        sanitized_data.append(
            {"input": sanitized_input, "sent_id": match["sent_id"], "output": match["output"]}
        )

    return sanitized_data


def get_unique_features(data):
    available_features = set()
    for match in data:
        for feat, value in match["input"].items():
            vals = value if isinstance(value, set) else {value}
            available_features.update(f"{feat}={value}" for value in vals)
    return sorted(available_features)


def build_feature_matrix(data, max_degree, min_feature_occurence):

    data_inputs = list()
    data_outputs = list()
    for sentence in data:
        data_inputs.append(sentence["input"])
        data_outputs.append(sentence["output"])

    print("Extracting features", flush=True)
    feature_set = FeatureSet()

    feature_set.add_feature(AllSingletonFeatures())
    for degree in range(2, max_degree + 1):
        feature_set.add_feature(
            AllProductFeatures(degree=degree, min_occurences=min_feature_occurence)
        )

    try:
        feature_set.init_from_data(data_inputs)
        all_feature_names = [name for fset in feature_set.features for name in fset.get_all_names()]
        X = feature_set.build_features(data_inputs, sparse=True)

        assert X.nnz == 0 or X.data.max() <= 1, f"values > 1 (max={X.data.max()})"

        if X.shape[1] == 0:
            raise RuntimeError("Empty feature list!")
    except RuntimeError:
        raise RuntimeError("There was an error during feature extraction")

    # TODO: this should be in features.pyx, somewhre
    col_counts = np.array((X != 0).sum(axis=0)).flatten()
    keep_features = np.where(col_counts >= min_feature_occurence)[0]
    X = X[:, keep_features]
    feature_names = [all_feature_names[i] for i in keep_features]

    labels = LabelEncoder()
    y = labels.fit_transform(data_outputs).astype(np.int_)

    return FeatureMatrix(X, y, feature_names, labels)


@dataclass
class FeatureMatrix:
    """Feature matrix for rule extraction.

    ``y`` holds the full set of encoded labels (0...k-1). If it's binary, the
    are built on a 0/1 view via `binarize`; ``labels`` (a fitted ``LabelEncoder``)
    maps encoded indices back to the original string labels.
    """

    X: spmatrix
    y: np.ndarray
    feature_names: list[tuple[str, ...]]
    labels: LabelEncoder

    @property
    def n_classes(self) -> int:
        return len(self.labels.classes_)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    @property
    def classes(self) -> list[str]:
        return list(self.labels.classes_)

    def binarize(self, positive: str) -> np.ndarray:
        """0/1 target for the binary core: 1 where label == positive, else 0."""
        if positive not in self.labels.classes_:
            raise ValueError(f"{positive!r} not in {self.classes}")
        pos = self.labels.transform([positive])[0]
        return (self.y == pos).astype(np.int_)


@dataclass
class ExtractionConfig:
    """Parsed yaml config."""

    scope: str
    conclusion: str | None
    predicate: FeaturePredicate
    positive: str = "Yes"

    @classmethod
    def from_yaml(cls, path: str) -> "ExtractionConfig":
        with open(path) as f:
            config = yaml.load(f, Loader=yaml.Loader)
        templates = FeaturePredicate.from_config(config["templates"])
        predicate = FeaturePredicate.from_config(config["features"], templates=templates)
        return cls(
            config["scope"], config.get("conclusion"), predicate, config.get("positive", "Yes")
        )


class CorpusData:
    """Extracted samples for some paths, a scope and a conclusion."""

    def __init__(
        self, samples: list[dict], scope: str, conclusion: str | None, positive: str = "Yes"
    ) -> None:
        self.samples = samples
        self.scope = scope
        self.conclusion = conclusion
        self.positive = positive

    @classmethod
    def from_corpus(
        cls, treebank_paths, config: ExtractionConfig, grew_config: str = "ud"
    ) -> "CorpusData":
        samples = extract_data(
            treebank_paths, config.scope, config.conclusion, config.predicate, grew_config
        )
        obj = cls(samples, config.scope, config.conclusion, config.positive)
        obj._validate()
        return obj

    def __len__(self) -> int:
        return len(self.samples)

    def _validate(self) -> None:
        # some checks
        if len(self.samples) == 0:
            raise RuntimeError("Patterns resulted in empty dataset")
        labels = {s["output"] for s in self.samples}
        if len(labels) < 2:
            raise RuntimeError("The conclusion is constant: only one class in the dataset")

    def unique_features(self) -> list[str]:
        return get_unique_features(self.samples)

    def to_matrix(self, max_degree: int, min_feature_occurence: int) -> FeatureMatrix:
        return build_feature_matrix(self.samples, max_degree, min_feature_occurence)
