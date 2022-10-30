import re


def read_log(log_file):
    database = list()
    compiler = re.compile(
        "(?P<status>.)( )+(?P<Expl>[0-9]+)( )+"
        + "(?P<Unexpl>[0-9]+) (?P<Obj>.{10})( ){1,4}"
        + "(?P<Depth>[0-9]*)( ){1,2}(?P<IntInf>[0-9]*)( )+"
        + "(?P<Incumbent>[0-9-.]+)[ ]+(?P<BestBd>[0-9.]+)[ ]+"
        + "(?P<Gap>[0-9.]*)[- %]{1,4}(?P<It_Node>[0-9]*)"
        + "([ -])+(?P<Time>[0-9]+)s"
    )
    for line in open(log_file):
        if (match := compiler.match(line)) is not None:
            database.append(match.groupdict())
    return database
