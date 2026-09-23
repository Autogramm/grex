import itertools
import collections
import numpy as np
import scipy.sparse

from grex.utils.utils import Dict


# Should not be used,
# I implemented this because unregularized intercept term
# was not implemented in celer. However, as I now use skglm,
# this is useless...
class InterceptFeature:
    def __init__(self):
        self.initialized = True

    def init_from_data(self, data):
        pass

    def build_features(self, X, data, offset):
        X[:, offset] = 1

    def get_all_names(self):
        return ["intercept"]

    def __len__(self):
        return 1


class ClassFeature:
    def __init__(self, name):
        self.name = name
        self.initialized = False

    def init_from_data(self, data):
        values = set()
        for sentence in data:
            if self.name in sentence:
                v = sentence[self.name]
                assert type(v) == str
                values.add(v)
        if len(values) == 0:
            raise RuntimeError("No value found for feature")
        self.dict = Dict(values)
        self.initialized = True

    def build_features(self, rows, cols, vals, data, offset):
        for i, dep in enumerate(data):
            if self.name in dep:
                j = self.dict.str_to_id(dep[self.name])
                rows.append(i)
                cols.append(offset + j)
                vals.append(1)

    def get_all_names(self):
        return [("%s=%s" % (self.name, v),) for v in self.dict._id_to_str]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            return len(self.dict)


class IndicatorFeature:
    def __init__(self, name):
        self.name = name
        self.initialized = False

    def init_from_data(self, data):
        values = set()
        for dep in data:
            if self.name in dep:
                v = dep[self.name]
                assert type(v) == set
                values.update(v)
        if len(values) == 0:
            raise RuntimeError("No value found for feature")
        self.dict = Dict(values)
        self.initialized = True

    def build_features(self, rows, cols, vals, data, offset):
        for i, dep in enumerate(data):
            if self.name in dep:
                for value in dep[self.name]:
                    j = self.dict.str_to_id(value)
                    rows.append(i); cols.append(offset + j); vals.append(1)

    def get_all_names(self):
        return [("%s=%s" % (self.name, v),) for v in self.dict._id_to_str]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return len(self.dict)


class AllSingletonFeatures:
    def __init__(self, predicate=None, min_occurences=1):
        self.predicate = predicate
        self.min_occurences = min_occurences
        self.initialized = False

    def init_from_data(self, data):
        class_feature_names = set()
        indicator_feature_names = set()
        for sentence in data:
            for k, v in sentence.items():
                if self.predicate is not None and not self.predicate(k):
                    continue
                if type(v) == str:
                    class_feature_names.add(k)
                elif type(v) == set:
                    indicator_feature_names.add(k)
                else:
                    raise RuntimeError("Unusable data type for feature %s: %s" % (k, type(v)))

        if len(class_feature_names.intersection(indicator_feature_names)) != 0:
            raise RuntimeError("Error in feature types")

        self.features = list()
        self.len_ = 0
        for name in sorted(class_feature_names):
            feature = ClassFeature(name)
            feature.init_from_data(data)
            self.len_ += len(feature)
            self.features.append(feature)

        for name in sorted(indicator_feature_names):
            feature = IndicatorFeature(name)
            feature.init_from_data(data)
            self.len_ += len(feature)
            self.features.append(feature)
        self.initialized = True

    def build_features(self, rows, cols, vals, data, offset):
        offset2 = 0
        for feature in self.features:
            feature.build_features(rows, cols, vals, data, offset + offset2)
            offset2 += len(feature)

    def get_all_names(self):
        return itertools.chain(*[feature.get_all_names() for feature in self.features])

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return self.len_


class AllProductFeatures:
    def __init__(self, degree=2, weight=1, min_occurences=1, predicate=None):
        self.predicate = predicate
        self.initialized = False
        self.degree = degree
        self.min_occurences = min_occurences
        self.weight = weight

    def init_from_data(self, data):
        class_feature_names = set()
        indicator_feature_names = set()
        for dep in data:
            for k, v in dep.items():
                if self.predicate is not None and not self.predicate(k):
                    continue
                if isinstance(v, str):
                    class_feature_names.add(k)
                elif isinstance(v, set):
                    indicator_feature_names.add(k)
                else:
                    raise RuntimeError("Unusable data type for feature %s: %s" % (k, type(v)))

        if class_feature_names & indicator_feature_names:
            raise RuntimeError("Error in feature types")

        # tpl = (bool, feat)
        merged_features = sorted(
            [(False, k) for k in class_feature_names]
            + [(True, k) for k in indicator_feature_names],
            key=lambda tpl: tpl[1],
        )

        feature_count = collections.defaultdict(int)
        for dep in data:
            existing_features = [(is_set, name) for (is_set, name) in merged_features if name in dep]
            for names in itertools.combinations(existing_features, self.degree):
                all_keys = [[]]
                for is_set_feature, name in names:
                    if is_set_feature:
                        all_keys = [k + [(name, v)] for k in all_keys for v in dep[name]]
                    else:
                        all_keys = [k + [(name, dep[name])] for k in all_keys]
                for key in all_keys:
                    feature_count[tuple(key)] += 1

        if self.min_occurences > 1:
            filtered_features = [k for k, v in feature_count.items() if v >= self.min_occurences]
        else:
            filtered_features = list(feature_count.keys())

        self.templates = dict()
        self.n_features = 0
        for feature in filtered_features:
            ks = tuple(k for k, _ in feature)
            vs = tuple(v for _, v in feature)
            if ks not in self.templates:
                self.templates[ks] = {vs: self.n_features}
            elif vs not in self.templates[ks]:
                self.templates[ks][vs] = self.n_features
            else:
                raise RuntimeError("SHOULD NOT HAPPEN")
            self.n_features += 1

        self.initialized = True
        self.valid_features = filtered_features

    def build_features(self, rows, cols, vals, data, offset):
        for i, dep in enumerate(data):
            for ks, ks_vs in self.templates.items():
                instance_values = []
                valid = True
                for k in ks:
                    if k not in dep:
                        valid = False
                        break
                    elif isinstance(dep[k], str):
                        instance_values = ([[dep[k]]] if not instance_values
                                        else [vs + [dep[k]] for vs in instance_values])
                    else:
                        instance_values = ([[v] for v in dep[k]] if not instance_values
                                        else [vs + [v] for vs in instance_values for v in dep[k]])
                if valid:
                    for vs in instance_values:
                        j = ks_vs.get(tuple(vs), -1)
                        if j >= 0:
                            rows.append(i)
                            cols.append(offset + j)
                            vals.append(self.weight)

    def get_all_names(self):
        return [tuple(f"{k}={v}" for k, v in feature) for feature in self.valid_features]

    def __len__(self):
        if not self.initialized:
            raise RuntimeError("Feature not initialized")
        else:
            # it always takes a single column
            return self.n_features


class FeatureSet:
    def __init__(self):
        self.features = list()

    def add_feature(self, feature):
        self.features.append(feature)

    def init_from_data(self, data):
        for feature in self.features:
            feature.init_from_data(data)

    def build_features(self, data, sparse=True):
        n_columns = sum(len(f) for f in self.features)
        rows, cols, vals = [], [], []
        offset = 0
        for feature in self.features:
            feature.build_features(rows, cols, vals, data, offset)
            offset += len(feature)
        X = scipy.sparse.coo_matrix(
            (vals, (rows, cols)), shape=(len(data), n_columns)
        )
        return X.tocsc() if sparse else X.toarray()

    def feature_weights(self, weights, ignore_zeros=True):
        ret = dict()
        offset = 0
        for feature in self.features:
            names = feature.get_all_names()
            for i, name in enumerate(names):
                if not ignore_zeros or not np.isclose(weights[offset], 0):
                    ret[name] = (weights[offset], offset)
                offset += 1
        return ret

    def print_weights(self, weights, ignore_zeros=True):
        for n, (v, _) in self.feature_weights(weights, ignore_zeros=ignore_zeros).items():
            print("%s:\t%.4f" % (n, v))
