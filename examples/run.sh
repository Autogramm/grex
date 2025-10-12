#!/bin/bash

python3 extract_rules_via_lasso.py \
examples/data/fr_pud-sud-test.conllu \
--config sud \
--patterns examples/patterns_subject_inversion.yaml \
--output examples/results/rules_subject_inversion_lasso.json

python3 extract_rules_via_dtree.py \
examples/data/fr_pud-sud-test.conllu \
--config sud \
--patterns examples/patterns_subject_inversion.yaml \
--output examples/results/rules_subject_inversion_dtree.json \
--only-leaves

python3 extract_rules_via_lasso.py \
examples/data/fr_pud-sud-test.conllu \
--config sud \
--patterns examples/patterns_agreement.yaml \
--output examples/results/rules_agreement.json \
--max-degree 2

python3 extract_rules_via_lasso.py \
examples/data/objects_spanish_romanian.conllu \
--patterns examples/patterns_contrastive.yaml \
--max-degree 2 \
--output examples/results/rules_contrastive_es_ro.json