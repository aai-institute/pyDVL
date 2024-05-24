from typing import Type


class NotFittedException(ValueError):
    def __init__(self, object_type: Type):
        super().__init__(
            f"Objects of type {object_type} must be fitted before calling "
            f"methods. "
            f"Call method fit with appropriate input."
        )
