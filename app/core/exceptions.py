class ApiKeyNotFoundError(Exception):
    def __init__(self, message="API key is missing."):
        super().__init__(message)