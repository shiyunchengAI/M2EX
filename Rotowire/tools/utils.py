def docstring(docstr, sep="\n"):
    def _decorator(func):
        if func.__doc__ is None:
            func.__doc__ = docstr
        else:
            func.__doc__ = sep.join([func.__doc__, docstr])
        return func
    return _decorator
