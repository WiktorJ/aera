"""Header as a plain attribute bag (see ../README.md)."""


class Header:
    def __init__(self, **kw):
        self.stamp = None
        self.frame_id = ""
        self.__dict__.update(kw)
