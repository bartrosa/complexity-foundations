"""Conway combinatorial games: core data structure."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Game:
    """Conway combinatorial game G = {L | R}.

    Recursive structure: left and right options are themselves Games.
    Frozen and hashable so games can be cached and used in sets.

    Reference: Conway 1976, On Numbers and Games, Chapter 0-2.
    """

    left: frozenset = field(default_factory=frozenset)
    right: frozenset = field(default_factory=frozenset)

    def __hash__(self):
        return hash((self.left, self.right))

    def __repr__(self):
        left_s = ",".join(map(str, sorted(self.left, key=hash))) if self.left else ""
        right_s = ",".join(map(str, sorted(self.right, key=hash))) if self.right else ""
        return f"{{{left_s} | {right_s}}}"
