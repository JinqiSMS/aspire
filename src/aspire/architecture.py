"""problem_setup_and_preliminaries.tex, sec:problem-prelim."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Architecture:
    d: int
    hidden_widths: tuple[int, ...]
    k: int = 4

    def __post_init__(self):
        object.__setattr__(self, "hidden_widths", tuple(self.hidden_widths))
        widths = (self.d,) + self.hidden_widths
        if len(widths) < 3 or any(type(n) is not int or n < 2 for n in widths):
            raise ValueError("Effective architecture requires L>=3 and all hidden widths>1")
        if any(a < b for a, b in zip(widths, widths[1:])):
            raise ValueError("Widths must be non-increasing")
        if type(self.k) is not int or self.k < 4 or self.k % 2:
            raise ValueError("The main algorithm requires even k>=4")

    @property
    def L(self): return len(self.hidden_widths) + 1

    @property
    def Q(self): return self.k ** (self.L - 1)

    @property
    def widths(self): return (self.d,) + self.hidden_widths

    def suffix_degree(self, layer):
        if not 1 <= layer < self.L: raise ValueError("Invalid one-based layer")
        return self.k ** (self.L - layer)

