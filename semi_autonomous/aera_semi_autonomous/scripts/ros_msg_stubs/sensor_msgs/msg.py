"""Image/JointState as plain attribute bags (see ../README.md)."""


class Image:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class JointState:
    def __init__(self, **kw):
        self.name = []
        self.position = []
        self.velocity = []
        self.effort = []
        self.header = None
        self.__dict__.update(kw)
