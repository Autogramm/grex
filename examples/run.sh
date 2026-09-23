#!/bin/bash

python3 ../src/grex/cli/extract_rules_via_dtree.py \
    ../tests/data/fr_pud-sud-test_10sents.conllu \
    --config sud \
    --patterns ../tests/data/patterns_test.yaml \
    --output results/test_dtree.json \
    --only-leaves


python3 ../src/grex/cli/extract_rules_via_lasso.py \
    ../tests/data/fr_pud-sud-test_10sents.conllu \
    --config sud \
    --patterns ../tests/data/patterns_test.yaml \
    --output results/test_lasso.json \
    --max-degree 2 \
    --alpha-num 5


# python3 ../src/grex/cli/extract_rules_via_lasso.py \
# data/fr_pud-sud-test.conllu \
# --config ud \
# --patterns patterns_subject_inversion.yaml \
# --output results/rules_subject_inversion_lasso.json \
# --max-degree 2 \
# --alpha-num 5


# python3 ../src/grex/cli/extract_rules_via_dtree.py \
# data/fr_pud-sud-test.conllu \
# --config ud \
# --patterns patterns_subject_inversion.yaml \
# --output results/rules_subject_inversion_dtree.json \
# --only-leaves


# python3 ../src/grex/cli/extract_rules_via_lasso.py \
# data/fr_pud-sud-test.conllu \
# --config ud \
# --patterns patterns_agreement.yaml \
# --output results/rules_agreement.json \
# --max-degree 2


# python3 extract_rules_via_lasso.py \
# examples/data/objects_spanish_romanian.conllu \
# --patterns examples/patterns_contrastive.yaml \
# --max-degree 2 \
# --output examples/results/rules_contrastive.json