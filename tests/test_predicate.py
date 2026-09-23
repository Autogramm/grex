from pathlib import Path

import grewpy

from grex.data.predicate import FeaturePredicate
from grex.data.data import CorpusData, ExtractionConfig

DATA_DIR = Path(__file__).parent / "data"


def test_predicate_include_excludes_correctly():
    pred = FeaturePredicate.from_config({
        "X": {
            "own": {"method": "include", "regexp": "upos=NOUN"},
            "parent": {"method": "exclude", "regexp": "xpos", "lemma_top_k": 5},
            "child": {"method": "include", "regexp": "upos=NOUN"},    
            },
        "sentence": {
            "meta": {"method": "include", "regexp": "sent_id"}
            }
    })

    assert all(m in ["X", "sentence"] for m in pred.matchers)
    assert pred("X", "own", "upos", "NOUN") is True
    assert pred("X", "own", "lemma", "dog") is False
    assert pred("X", "parent", "form", "est") is True
    assert pred("X", "parent", "lemma", "être") is True
    assert pred("X", "parent", "xpos") is False
    assert pred("X", "own", "upos", "VERB") is False
    assert pred("sentence", "meta", "sent_id", "n01003013") is True


def test_matrix_column_matches_data_count():
    config = ExtractionConfig.from_yaml(str(DATA_DIR / "patterns_test.yaml"))
    cd = CorpusData.from_corpus(str(DATA_DIR / "fr_pud-sud-test_10sents.conllu"), config)
    fm = cd.to_matrix(max_degree=1, min_feature_occurence=1)
    target = ("node:Y:own:upos=NOUN",)
    assert len(cd.samples) == 14
    corpus = grewpy.Corpus(str(DATA_DIR / "fr_pud-sud-test_10sents.conllu"))
    from_grew = corpus.count(grewpy.Request("pattern { X-[1=subj]->Y; Y[upos=NOUN]}"))
    from_data = sum(
        1 for s in cd.samples 
        if s["input"].get("node:Y:own:upos") == "NOUN"
    )
    col = fm.feature_names.index(target)
    from_matrix = int(fm.X[:, col].sum())
    assert from_data == from_matrix == from_grew
