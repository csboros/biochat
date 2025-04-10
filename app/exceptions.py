class BusinessException(Exception):
    """Base exception class for business logic errors that should be displayed to users."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
