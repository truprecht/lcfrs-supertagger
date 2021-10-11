from discodop.tree import Tree


def float_or_zero(s):
    try:
        f = float(s)
        return f if f == f else 0.0 # return 0 if f is NaN
    except:
        return 0.0


def parse_or_none(s: str, parse_t = str):
    if s == "None":
        return None
    if parse_t is str:
        return s
    return parse_t(s)


def noparse(partial: Tree, postags: list) -> Tree:
    if partial is None:
        return Tree("NOPARSE", [Tree(pt, [i]) for i, pt in enumerate(postags)])
    missing = set(range(len(postags))) - set(partial.leaves())
    if not missing:
        return partial
    return Tree(
        "NOPARSE",
        [partial] + [ Tree(postags[i], [i]) for i in missing ]
    )
