import grewpy
import re

CLOSED_POS_TAGS = {
    "AUX",
    "ADP",
    "PRON",
    "DET",
    "SCONJ",
    "CCONJ",
    "PART"
}

# le terme "similar" n'est pas du tout adapté,
# mais c'est en gros les POS tags qu'on veut merger ensemble
# pour ajouter de nouvelles features
# càd que plutôt qu'avoir seulement "POS=NOUN",
# on veut aussi avoir "POS=NOUN|PROPN"
SIMILAR_POS_TAGS = [
    # use list to fix order, probably unecessary,
    # but just in case
    [
        "NOUN", "PROPN"
    ],
    [
        "NOUN", "PROPN", "PRON"
    ],
    [
        "AUX", "VERB"
    ],
    [
        "DET", "NUM"
    ]
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
        if predicate(node_name, relation_name, k):
            if k == "lemma":
                # for lemmas we need to keep the upos information for filtering
                ret[("node", node_name, relation_name, k)] = (v, features["upos"])
            else:
                ret[("node", node_name, relation_name, k)] = v
    return ret


def extract_features(draft, match, feature_predicate, include_metadata=False):
    sentence = draft[match["sent_id"]]
    parents = parents_from_successors(sentence.sucs)
    selected_node_ids = set(match["matching"]["nodes"].values())
    features = dict()

    # sentence level meta features
    if include_metadata:
        for k, v in sentence.meta.items():
            if feature_predicate('sentence','meta', k):
                features[("meta", k)] = v

    for node_name, node_id in match["matching"]["nodes"].items():

        if node_id == "0": # fake root
            continue
        
        # node features
        features.update(build_node_features(
            node_name,
            "own",
            sentence.features[node_id],
            feature_predicate
        ))

        # node relation
        parent_id, rel = parents[node_id]
        if feature_predicate(node_name, "own", "rel_shallow"):
            k = ("node", node_name, "own", "rel_shallow")
            v = ":".join(rel[rel_key] for rel_key in ["1", "2"] if rel_key in rel)
            features[k] = v
        if feature_predicate(node_name, "own", "rel_deep") and "deep" in rel:
            k = ("node", node_name, "own", "rel_deep")
            v = rel['deep']
            features[k] = v

        # position of parent
        if feature_predicate(node_name, "parent", "position"):
            features[("node", node_name, "parent", "position")] = "before" if int(parent_id) < int(node_id) else "after"

        # parent features
        if parent_id not in selected_node_ids:
            features.update(build_node_features(
                node_name,
                "parent",
                sentence.features[parent_id],
                feature_predicate
            ))

        # prev word
        prev_id = str(int(node_id) - 1)
        if prev_id in sentence.features and prev_id not in selected_node_ids:
            features.update(build_node_features(
                node_name,
                "prev",
                sentence.features[prev_id],
                feature_predicate
            ))

        # next word
        next_id = str(int(node_id) + 1)
        if next_id in sentence.features and next_id not in selected_node_ids:
            features.update(build_node_features(
                node_name,
                "next",
                sentence.features[next_id],
                feature_predicate
            ))

        # children features
        for child_id, child_rel in sentence.sucs.get(node_id, list()):
            # only consider children that are not matched nodes
            if child_id in selected_node_ids:
                continue

            # child features
            child_features = sentence.features[child_id]
            for k, v in child_features.items():
                if feature_predicate(node_name, "child", k):
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
            if feature_predicate(node_name, "child", "rel_shallow"):
                k = ("node", node_name, "child", "rel_shallow")
                v = ":".join(rel[rel_key] for rel_key in ["1", "2"] if rel_key in rel)
                if k not in features:
                    features[k] = {v}
                else:
                    features[k].add(v)
            if feature_predicate(node_name, "child", "rel_deep") and "deep" in rel:
                k = ("node", node_name, "child", "rel_deep")
                v = rel["deep"]
                if k not in features:
                    features[k] = {v}
                else:
                    features[k].add(v)

    return features


def extract_data(treebank_paths, scope, conclusion, conclusion_meta, feature_predicate, config="ud", include_metadata=False):
    grewpy.set_config(config)

    if conclusion is None and conclusion_meta is None:
        raise RuntimeError("No conclusion provided in configuration")

    if type(treebank_paths) == str:
        treebank_paths = [treebank_paths]

    data = []
    for tp in treebank_paths:
        corpus = grewpy.Corpus(tp)
        draft = grewpy.CorpusDraft(corpus)

        req = grewpy.Request(scope)
        if conclusion is not None:
            matches = corpus.search(req, clustering_parameter=["{" + conclusion + "}"])
            matches = [(sent, c, sent["sent_id"]) for c, sents in matches.items() for sent in sents]
        else:
            matches = [(sent, "Yes", sent["sent_id"]) for sent in corpus.search(req)]

        if conclusion_meta is not None:
            conclusion_meta = {
                k: v if type(v) is list else [v]
                for k, v in conclusion_meta.items()
            }

            matches = [
                (sent, "No", sent["sent_id"])
                if c == "No"
                else (
                    (sent, "Yes")
                    if all(
                        any(re.fullmatch(p, draft[sent['sent_id']].meta[k]) for p in v)
                        for k, v in conclusion_meta.items()
                    )
                    else (sent, "No")
                )
                for sent, c in matches
                if all(k in draft[sent['sent_id']].meta for k in conclusion_meta.keys())
            ]

        for match, c, sent_id in matches:
            assert c in ["Yes", "No"]

            data.append({
                "input": extract_features(draft, match, feature_predicate, include_metadata),
                "sent_id": sent_id,
                "output": 1 if c == "Yes" else 0
            })

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
                feature_predicate.update_lemmas_counter(node_name, relation_name, feature_value[0], feature_value[1])
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
                    if not feature_predicate.check_lemma(k[1], k[2], feature_value[0], feature_value[1]):
                        continue
                    feature_value = feature_value[0]
            sanitized_input[":".join(k)] = feature_value

        sanitized_data.append({
            "input": sanitized_input,
            "sent_id": match["sent_id"],
            "output": match["output"]
        })

    return sanitized_data