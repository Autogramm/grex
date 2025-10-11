import re
import collections

from grewpy import Request

ALLOWED_FEATURE_POSITIONS = ["own", "parent", "child", "prev", "next", "meta"]
ALLOWED_KEYS = {"method", "regexp", "lemma_top_k", "lemma_upos_filter"}
REQUIRED_KEYS = {"method", "regexp"}

class Dict:
    def __init__(self, values):
        values = set(values)
        self._id_to_str = list()
        self._str_to_id = dict()

        for v in values:
            self._str_to_id[v] = len(self._id_to_str)
            self._id_to_str.append(v)

    def str_to_id(self, v):
        return self._str_to_id[v]

    def id_to_str(self, v):
        return self._id_to_str[v]

    def __len__(self):
        return len(self._id_to_str)


class StringMatcher:
    def __init__(self, method, regexps):
        assert method in ["include", "exclude"]
        self.include = (method == "include")

        if type(regexps) != list:
            regexps = [regexps]
        assert all(type(p) == str for p in regexps)
        self.regexps = regexps

    def __call__(self, string):
        m = any(re.fullmatch(p, string) for p in self.regexps)
        return m if self.include else not m


class LemmaFilter:
    def __init__(self, top_k=0, allowed_upos=list()):
        self.top_k = top_k
        self.allowed_upos = allowed_upos
        self.counter = None
        self.is_initialized = False
        self.allowed_lemmas = None

    def check_initialization(self):
        if not self.is_initialized:
            raise RuntimeError("Unitialized")

    def transform_upos(self, upos):
        if len(self.allowed_upos) == 0:
            return "**UNDEF**"
        else:
            return upos

    def transform_lemma(self, lemma):
        return lemma.lower()

    def reset_counter(self):
        self.counter = collections.Counter()
        self.is_initialized = False

    def freeze_counter(self):
        assert self.counter is not None
        self.is_initialized = True
        if self.top_k <= 0:
            self.allowed_lemmas = None
        else:
            allowed_lemmas = collections.defaultdict(lambda: set())
            for (lemma, upos), c in self.counter.most_common(self.top_k):
                allowed_lemmas[upos].add(lemma)
            self.allowed_lemmas = dict(allowed_lemmas)

    def update_counter(self, lemma, upos):
        assert self.counter is not None
        upos = self.transform_upos(upos)
        lemma = self.transform_lemma(lemma)

        if len(self.allowed_upos) == 0 or upos in self.allowed_upos:
            self.counter[(lemma, upos)] += 1

    def __call__(self, lemma, upos):
        self.check_initialization()

        if self.top_k < 0:
            return True
        if self.top_k == 0:
            return False

        upos = self.transform_upos(upos)
        lemma = self.transform_lemma(lemma)
        return lemma in self.allowed_lemmas.get(upos, set())


class FeaturePredicate:
    def __init__(self):
        self.matchers = dict()
        self.lemma_filters = dict()

    @staticmethod
    def from_config(config, templates=dict()):
        obj = FeaturePredicate()

        for node, tpl in config.items():
            if type(tpl) == str:
                obj.matchers[node] = templates.matchers[tpl]
                obj.lemma_filters[node] = templates.lemma_filters[tpl]
            else:
                assert node not in obj.matchers
                obj.matchers[node] = dict()
                obj.lemma_filters[node] = dict()
                for k, v in tpl.items():
                    assert k in ALLOWED_FEATURE_POSITIONS
                    v_keys = set(v.keys())
                    assert REQUIRED_KEYS.issubset(v_keys), f"Missing required keys in '{k}'. Must contain {REQUIRED_KEYS}"
                    assert v_keys.issubset(ALLOWED_KEYS), f"Unexpected keys in '{k}'. Allowed keys are {ALLOWED_KEYS}"
                    obj.matchers[node][k] = StringMatcher(v["method"], v["regexp"])
                    obj.lemma_filters[node][k] = LemmaFilter(v.get("lemma_top_k", -1), v.get("lemma_upos_filter", list()))

        return obj

    def __call__(self, name, where, feature):
        assert where in ALLOWED_FEATURE_POSITIONS
        if name not in self.matchers:
            raise KeyError("Feature matching has not been implemented for node '%s'" % name)
        if where not in self.matchers[name]:
            return False
        else:
            return self.matchers[name][where](feature)

    def reset_lemmas_counter(self):
        for k, v in self.lemma_filters.items():
            for k2, v2 in v.items():
                v2.reset_counter()

    def freeze_lemmas_counter(self):
        for k, v in self.lemma_filters.items():
            for k2, v2 in v.items():
                v2.freeze_counter()

    def update_lemmas_counter(self, node_name, rel_name, lemma, upos):
        self.lemma_filters[node_name][rel_name].update_counter(lemma, upos)

    def check_lemma(self, node_name, rel_name, lemma, upos):
        return self.lemma_filters[node_name][rel_name](lemma, upos)

def pattern_to_request(pattern, scope):
    """
    Build a Grew request from a Grex pattern
    """
    def parents_in_scope(scope: str) -> dict:
        """Get scope dependencies. Parent relations are needed to build a grew request."""
        parents = dict()
        for clause in Request(scope).json_data():
            for constraint in clause['pattern']: # type: ignore
                if "->" in constraint:
                    parent, child = re.split("-.*-?>", constraint)
                    parents[child] = parent
        return parents

    scope_parents = parents_in_scope(scope)
    request = Request(scope)

    for att in pattern:
        if att.startswith("0") or att.startswith("1"): # dtree rules contain the split decision
            sign, _, node_name, target, feature = att.split(":", maxsplit=4)
            if int(sign):
                keyword = "with" if target == "child" else "pattern"
            else:
                keyword = "without"
        else:
            _, node_name, target, feature = att.split(":", maxsplit=3)
            keyword = "pattern"

        feat, value = feature.split("=")
        parent = scope_parents.get(node_name, f"{node_name}parent")

        # position
        if feat == "position":
            if value == "after":
                request.append(keyword, f"{parent} << {node_name}")
            else:
                request.append(keyword, f"{node_name} << {parent}")
    
        # deprels
        elif "rel_shallow" in feat:
            deprel = value.split(":")
            rel = f"1={deprel[0]}, 2={deprel[1]}" if len(deprel) == 2 else f"1={value}"
            if target == "own":
                request.append(keyword, f'{parent}-[{rel}]->{node_name}')
            else: #child
                request.append(keyword, f'{node_name}-[{rel}]->{node_name}child')
        elif "rel_deep" in feat:
            if target == "own":
                request.append(keyword, f'{parent}-[deep={value}]->{node_name}')
            else: #child
                request.append(keyword, f'{node_name}-[deep={value}]->{node_name}child')
    
        # features
        elif target == "prev":
            if sign and any(f'{node_name}prev<{node_name}' in str(item) and 'pattern' in str(item) for item in request.items): 
                request.append(keyword, f'{node_name}{target}[{feat}="{value}"]')
            else:
                request.append(keyword, f'{node_name}prev<{node_name}; {node_name}prev[{feat}="{value}"]')

        elif target == "next":
            if sign and any(f'{node_name}<{node_name}next' in str(item) and 'pattern' in str(item) for item in request.items): 
                request.append(keyword, f'{node_name}{target}[{feat}="{value}"]')
            else:
                request.append(keyword, f'{node_name}<{node_name}next; {node_name}next[{feat}="{value}"]')

        elif target == "child":
            request.append(keyword, f'{node_name}->{node_name}child; {node_name}child[{feat}="{value}"]')
        elif target == "parent":
            request.append(keyword, f'{parent}->{node_name}; {parent}[{feat}="{value}"]')
        else: #own
            request.append(keyword, f'{node_name}[{feat}="{value}"]')
            
    return request