"""
Custom exceptions for the LOMA Dataset package.
"""


class LomaDatasetError(Exception):
    """Base exception for all LOMA Dataset errors."""
    pass


class DatabaseError(LomaDatasetError):
    """Raised when database operations fail."""
    pass


class ProcessingError(LomaDatasetError):
    """Raised when dataset processing fails."""
    pass


class ValidationError(LomaDatasetError):
    """Raised when data validation fails."""
    pass


class ConfigurationError(LomaDatasetError):
    """Raised when configuration is invalid."""
    pass