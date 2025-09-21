import collections

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks"])
