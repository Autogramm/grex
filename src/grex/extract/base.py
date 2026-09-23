from abc import ABC, abstractmethod

import numpy as np
import scipy


class Extractor(ABC):
    def __init__(self, scope, conclusion):
        self.scope = scope
        self.conclusion = conclusion
        self.n_samples = None
        self.num_positive = None
        self._rules = []
        self._matches = []  # aligned with self._rules; each is a 1 x n_samples sparse boolean row

    @abstractmethod
    def fit(self, X, y, feature_names) -> None:
        self.n_samples = len(y)
        self.num_positive = int(np.sum(y))

    @abstractmethod
    def rules(self) -> list[dict]: ...

    def matches(self) -> scipy.sparse :
        """
        A sparse boolean matrix n_rules x n_samples
        Row i corresponds to rules()[i]: column s is True when rule i fires on sample s. 
        This keeps track of matches and is used for overlap analysis.
        """
        n = self.n_samples or 0
        if not self._matches:
            return scipy.sparse.csr_matrix((0, n), dtype=bool)
        assert len(self._matches) == len(self._rules), "matches/rules misaligned"
        return scipy.sparse.vstack(self._matches).tocsr()

    def save_matches(self, path) -> None:
        """
        Save the match matrix as .npz.
        Row i corresponds to rules()[i], so the rule JSON from to_dict() is the row index.
        """
        scipy.sparse.save_npz(path, self.matches())

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "conclusion": self.conclusion,
            "s_occs": self.n_samples,
            "q_occs": self.num_positive,
            "rules": self.rules(),
        }


def rule_statistics(n_matched, n_pos_in_p, n_neg_in_p, num_positive, n_total):
    """The stats block shared verbatim by lasso and dtree."""
    mu = num_positive / n_total
    a = n_pos_in_p / n_matched
    gstat = (
        2
        * n_matched
        * (
            ((a * np.log(a)) if a > 0 else 0)
            - a * np.log(mu)
            + (((1 - a) * np.log(1 - a)) if (1 - a) > 0 else 0)
            - (1 - a) * np.log(1 - mu)
        )
    )
    expected = (n_matched * num_positive) / n_total
    return {
        "coverage_q_in_p": n_pos_in_p / n_matched,
        "coverage_not_q_in_p": n_neg_in_p / n_matched,
        "delta": n_pos_in_p - expected,
        "g-statistic": gstat,
        "p-value": 1 - scipy.stats.chi2.cdf(gstat, 1),
        "cramers_phi": np.sqrt(gstat / n_matched),
    }
