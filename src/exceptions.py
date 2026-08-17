class UnavailableModelError(Exception):
    def __init__(self, *args):
        self.message = args[0] if args else None

    def __str__(self):
        if self.message:
            return f'{self.__class__.__name__} {self.message}'
        else:
            return f'{self.__class__.__name__} has been raised'
