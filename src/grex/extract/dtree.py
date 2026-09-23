import numpy as np
from sklearn import tree

from grex.extract.base import Extractor, rule_statistics
from grex.utils.request import pattern_to_request


def parents_from_dtree(T):
    """Build a dictionary child: (parent, sign) where sign is the split decision (0,1)."""
    parents = dict()
    for node in range(T.node_count):
        left = T.children_left[node]
        right = T.children_right[node]
        if left >= 0:
            parents[left] = (node, 0)
        if right >= 0:
            parents[right] = (node, 1)
    return parents


def branch_from_parents(node, parents):
    """Compute the branch from node n to the root of the decision tree."""
    branch = []
    while node != 0:
        node, split = parents[node]  # split = 0, 1 (resp. left, right)
        branch.append((node, split))
    branch.reverse()
    return branch


def pattern_from_dtree(T, branch, feature_names):
    """Build a rule pattern from a branch of the decision tree."""
    attributs = [(n, feature_names[T.feature[n]][0], decision) for n, decision in branch]
    pattern = [f"{decision}:{att}" for _, att, decision in attributs]
    return pattern


class DTreeExtractor(Extractor):
    def __init__(
        self,
        scope,
        conclusion,
        *,
        min_samples_leaf=5,
        max_depth=12,
        threshold=1e-1,
        only_leaves=True,
        grew=False,
        random_state=42,
    ):

        super().__init__(scope, conclusion)
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.threshold = threshold
        self.only_leaves = only_leaves
        self.grew = grew
        self.random_state = random_state
        self._rules = []

    def fit(self, X, y, feature_names):

        super().fit(X, y, feature_names)

        clf = tree.DecisionTreeClassifier(
            criterion="entropy",
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )

        clf.fit(X, y)
        T = clf.tree_
        dtree_parents = parents_from_dtree(T)
        node_indicator = clf.decision_path(X)        # (n_samples x n_nodes) sparse

        # which tree nodes become rules
        selected_nodes = []
        if self.only_leaves:
            for n in range(T.node_count):
                if T.children_left[n] == T.children_right[n] == -1:  # leaf
                    selected_nodes.append((int(np.argmax(T.value[n])), n))
        else:
            for n in range(T.node_count):
                if T.impurity[n] < self.threshold:
                    selected_nodes.append((1 if np.argmax(T.value[n]) else 0, n))

        rules = []
        match_rows = []
        for decision, n in selected_nodes:
            branch = branch_from_parents(n, dtree_parents)
            pattern = pattern_from_dtree(T, branch, feature_names)
            rule = pattern_to_request(pattern, self.scope) if self.grew else pattern

            n_matched = int(T.n_node_samples[n])
            n_pos = n_matched * T.value[n][0, 1]
            n_neg = n_matched * T.value[n][0, 0]

            stats = rule_statistics(n_matched, n_pos, n_neg, self.num_positive, self.n_samples)
            cov_p, cov_not_p = stats["coverage_q_in_p"], stats["coverage_not_q_in_p"]

            if decision:
                coverage = (n_pos / self.num_positive) * 100
                precision = (n_pos / n_matched) * 100
                ratio = (cov_p / cov_not_p) if (cov_p != 0 and cov_not_p != 0) else 0
            else:
                coverage = (n_neg / (self.n_samples - self.num_positive)) * 100
                precision = (n_neg / n_matched) * 100
                ratio = (cov_not_p / cov_p) if (cov_p != 0 and cov_not_p != 0) else 0

            rules.append(
                {
                    "pattern": rule,
                    "p_occs": n_matched,
                    "p_q_occs": int(n_pos),
                    "p_notq_occs": int(n_neg),
                    "decision": "yes" if decision else "no",
                    "coverage": coverage,
                    "coverage_q_in_p": cov_p,
                    "coverage_not_q_in_p": cov_not_p,
                    "precision": precision,
                    "ratio_coverage_in_p": ratio,
                    "delta": stats["delta"],
                    "g-statistic": stats["g-statistic"],
                    "p-value": stats["p-value"],
                    "cramers_phi": stats["cramers_phi"],
                }
            )
            match_rows.append((node_indicator[:, n] > 0).T.tocsr())

        self._rules = rules[::-1]  # reserve order
        self._matches = match_rows[::-1]

    def rules(self):
        return self._rules
