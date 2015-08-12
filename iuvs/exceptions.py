class IUVSError(Exception):
    pass


class DimensionsError(ValueError, IUVSError):
    pass


class UnknownEnvError(IUVSError):

    def __init__(self, env):
        self.env = env

    def __str__(self):
        return ("Given environment unknown: {}".format(self.env))


class PathNotReadableError(IUVSError):

    def __init__(self, path):
        self.path = path

    def __str__(self):
        return ("Provided path not readable:\n{}".format(self.path))
