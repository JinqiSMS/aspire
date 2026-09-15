"""Structured numerical failures are scientific output, not hidden fallbacks."""

class NumericalFailure(RuntimeError):
    def __init__(self, reason: str, **details):
        super().__init__(reason)
        self.reason = reason
        self.details = details

