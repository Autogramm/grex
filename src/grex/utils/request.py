import re

from grewpy import Request


def pattern_to_request(pattern, scope):
    """
    Build a Grew request from a Grex pattern
    """

    def parents_in_scope(scope: str) -> dict:
        """Get scope dependencies. Parent relations are needed to build a grew request."""
        parents = dict()
        for clause in Request(scope).json_data():
            for constraint in clause["pattern"]:  # type: ignore
                if "->" in constraint:
                    parent, child = re.split("-.*-?>", constraint)
                    parents[child] = parent
        return parents

    scope_parents = parents_in_scope(scope)
    request = Request(scope)

    for att in pattern:
        if att.startswith("0") or att.startswith("1"):  # dtree rules contain the split decision
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
                request.append(keyword, f"{parent}->{node_name}; {node_name} << {parent}")
            else:
                request.append(keyword, f"{parent}->{node_name}; {parent} << {node_name}")

        # deprels
        elif "rel_shallow" in feat:
            deprel = value.split(":")
            rel = f"1={deprel[0]}, 2={deprel[1]}" if len(deprel) == 2 else f"1={value}, !2"
            if target == "own":
                request.append(keyword, f"{parent}-[{rel}]->{node_name}")
            else:  # child
                request.append(keyword, f"{node_name}-[{rel}]->{node_name}child")
        elif "rel_deep" in feat:
            if target == "own":
                request.append(keyword, f"{parent}-[deep={value}]->{node_name}")
            else:  # child
                request.append(keyword, f"{node_name}-[deep={value}]->{node_name}child")

        # features
        elif target == "prev":
            if sign and any(
                f"{node_name}prev<{node_name}" in str(item) and "pattern" in str(item)
                for item in request.items
            ):
                request.append(keyword, f'{node_name}{target}[{feat}="{value}"]')
            else:
                request.append(
                    keyword, f'{node_name}prev<{node_name}; {node_name}prev[{feat}="{value}"]'
                )

        elif target == "next":
            if sign and any(
                f"{node_name}<{node_name}next" in str(item) and "pattern" in str(item)
                for item in request.items
            ):
                request.append(keyword, f'{node_name}{target}[{feat}="{value}"]')
            else:
                request.append(
                    keyword, f'{node_name}<{node_name}next; {node_name}next[{feat}="{value}"]'
                )

        elif target == "child":
            request.append(
                keyword, f'{node_name}->{node_name}child; {node_name}child[{feat}="{value}"]'
            )
        elif target == "parent":
            request.append(keyword, f'{parent}->{node_name}; {parent}[{feat}="{value}"]')
        else:  # own
            request.append(keyword, f'{node_name}[{feat}="{value}"]')

    return request
