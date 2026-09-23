import numpy as np
import skglm
from sklearn.linear_model import LogisticRegression

from grex.extract.base import Extractor, rule_statistics


class LassoExtractor(Extractor):
    def __init__(
        self,
        scope,
        conclusion,
        *,
        alpha_start=0.1,
        alpha_end=0.001,
        alpha_num=100,
        max_iter=20,
        max_epochs=1000,
    ):
        super().__init__(scope, conclusion)
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.alpha_num = alpha_num
        self.max_iter = max_iter
        self.max_epochs = max_epochs
        self.intercepts = []
        self._rules = []

    def fit(self, X, y, feature_names):
        super().fit(X, y, feature_names)

        all_rules = set()
        ordered_rules = []
        match_rows = []
        alphas = np.linspace(self.alpha_start, self.alpha_end, self.alpha_num)
        for j, alpha in enumerate(alphas):
            print("extracting rules (%i / %i)" % (j + 1, len(alphas)), flush=True)
            model = skglm.SparseLogisticRegression(
                alpha=alpha,
                fit_intercept=True,
                max_iter=self.max_iter,
                max_epochs=self.max_epochs,
            )
            model.fit(X, y)
            self.intercepts.append((alpha, model.intercept_))

            for idx, value in enumerate(model.coef_[0]):
                if value == 0:
                    continue
                name = feature_names[idx]
                if name in all_rules:  # record each feature once, at first non-zero alpha
                    continue
                all_rules.add(name)

                col = np.asarray(X[:, idx].todense())
                idx_col = col.squeeze(1)
                matched = y[idx_col > 0]
                n_matched = len(matched)
                n_pos = matched.sum()
                n_neg = n_matched - n_pos

                stats = rule_statistics(n_matched, n_pos, n_neg, self.num_positive, self.n_samples)
                cov_q = stats["coverage_q_in_p"]
                cov_not_q = stats["coverage_not_q_in_p"]

                if n_pos / n_matched > int(y.sum()) / self.n_samples:
                    decision = "yes"
                    coverage = (n_pos / self.num_positive) * 100
                    precision = (n_pos / n_matched) * 100
                    ratio = cov_q / cov_not_q if cov_not_q != 0 else 0
                else:
                    decision = "no"
                    coverage = (n_neg / (self.n_samples - self.num_positive)) * 100
                    precision = (n_neg / n_matched) * 100
                    ratio = cov_not_q / cov_q if cov_q != 0 else 0

                ordered_rules.append(
                    {
                        "pattern": name,  # tuple -> JSON array (not str())
                        "p_occs": int(idx_col.sum()),
                        "p_q_occs": int(n_pos),
                        "p_notq_occs": int(n_neg),
                        "decision": decision,
                        "lambda": alpha,
                        "coef": value,
                        "coverage": coverage,
                        "coverage_q_in_p": cov_q,
                        "coverage_not_q_in_p": cov_not_q,
                        "precision": precision,
                        "ratio": ratio,
                        "delta": stats["delta"],
                        "g-statistic": stats["g-statistic"],
                        "p-value": stats["p-value"],
                        "cramers_phi": stats["cramers_phi"],
                    }
                )
                match_rows.append((X[:, idx] > 0).T.tocsr())

        self._rules = ordered_rules
        self._matches = match_rows

    def rules(self):
        return self._rules

    def to_dict(self) -> dict:
        # override to insert intercepts, preserving original key order
        d = super().to_dict()

        return {
            "scope": d["scope"],
            "conclusion": d["conclusion"],
            "s_occs": d["s_occs"],
            "q_occs": d["q_occs"],
            "intercepts": self.intercepts,
            "rules": d["rules"],
        }


class MulticlassLassoExtractor(Extractor):
    """Multinomial L1 logistic regression (scikit-learn, SAGA solver) for >2 classes.

    A single joint softmax model over all classes with an L1 penalty (not
    one-vs-rest). For each class k, emits the features with non-zero weight in
    coef_[k] as rules (decision = class name), with coverage/precision computed
    treating k as the positive class. coef_[k] is class k's weight in the joint
    model, so a feature may characterize more than one class; rules are
    deduplicated on (feature, class).
    """

    def __init__(
        self, scope, conclusion, classes, *,
        C_start=1e-3, C_end=1e-1, C_num=100, max_iter=10000, only_positive=True,
    ):
        super().__init__(scope, conclusion)
        self.classes = list(classes) # index == encoded label
        self.C_start = C_start
        self.C_end = C_end
        self.C_num = C_num
        self.max_iter = max_iter
        self.only_positive = only_positive
        self.intercepts = []
        self.class_counts = None

    def fit(self, X, y, feature_names):
        super().fit(X, y, feature_names) # sets n_samples (num_positive unused here)
        self.class_counts = {k: int((y == k).sum()) for k in range(len(self.classes))}

        all_rules = set() # dedup on (feature_name, class)
        ordered_rules = []
        match_rows = []

        Cs = np.geomspace(self.C_start, self.C_end, self.C_num)
        for j, C in enumerate(Cs):
            print("extracting rules (%i / %i)" % (j + 1, len(Cs)), flush=True)
            model = LogisticRegression(
                penalty="l1", C=C, solver="saga", fit_intercept=True,
                max_iter=self.max_iter, warm_start=True, n_jobs=-1,
                class_weight="balanced",
            )
            model.fit(X, y)
            self.intercepts.append((C, model.intercept_))

            for row in range(model.coef_.shape[0]):
                k = int(model.classes_[row])
                num_pos_k = self.class_counts[k]
                coefs = model.coef_[row]
                idxs = np.where(coefs > 0)[0] if self.only_positive else np.where(coefs != 0)[0]

                for idx in idxs:
                    idx = int(idx)
                    name = feature_names[idx]
                    if (name, k) in all_rules:
                        continue
                    all_rules.add((name, k))

                    col = np.asarray(X[:, idx].todense()).squeeze(1)
                    matched_mask = col > 0
                    n_matched = int(matched_mask.sum())
                    if n_matched == 0:
                        continue
                    n_pos = int((y[matched_mask] == k).sum())
                    n_neg = n_matched - n_pos

                    stats = rule_statistics(n_matched, n_pos, n_neg, num_pos_k, self.n_samples)
                    cov_q = stats["coverage_q_in_p"]
                    cov_not_q = stats["coverage_not_q_in_p"]

                    ordered_rules.append(
                        {
                            "pattern": name,
                            "p_occs": int(col.sum()),
                            "p_q_occs": n_pos,
                            "p_notq_occs": n_neg,
                            "decision": self.classes[k],
                            "lambda": C,
                            "coef": float(coefs[idx]),
                            "coverage": (n_pos / num_pos_k) * 100 if num_pos_k else 0,
                            "coverage_q_in_p": cov_q,
                            "coverage_not_q_in_p": cov_not_q,
                            "precision": (n_pos / n_matched) * 100,
                            "ratio": cov_q / cov_not_q if cov_not_q != 0 else 0,
                            "delta": stats["delta"],
                            "g-statistic": stats["g-statistic"],
                            "p-value": stats["p-value"],
                            "cramers_phi": stats["cramers_phi"],
                        }
                    )
                    match_rows.append((X[:, idx] > 0).T.tocsr())

        self._rules = ordered_rules
        self._matches = match_rows

    def rules(self):
        return self._rules

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "conclusion": self.classes, # list of class names
            "s_occs": self.n_samples,
            "q_occs": [self.class_counts[k] for k in range(len(self.classes))],
            "intercepts": self.intercepts,
            "rules": self._rules,
        }
