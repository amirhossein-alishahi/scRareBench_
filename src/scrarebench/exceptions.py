class ScRareBenchError(Exception):
    """Base package exception."""


class DatasetValidationError(ScRareBenchError):
    """Raised when the benchmark dataset does not match the expected contract."""


class LatentAlignmentError(ScRareBenchError):
    """Raised when latent rows cannot be safely aligned to dataset cells."""


class MissingDependencyError(ScRareBenchError):
    """Raised when an optional runtime dependency is unavailable."""
