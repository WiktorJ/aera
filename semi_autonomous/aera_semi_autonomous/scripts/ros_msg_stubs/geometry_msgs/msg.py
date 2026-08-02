"""Pose/Point/Quaternion as plain attribute bags (see ../README.md)."""


class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)


class Pose:
    def __init__(self, position=None, orientation=None):
        self.position = position or Point()
        self.orientation = orientation or Quaternion()


class PoseStamped:
    def __init__(self):
        self.pose = Pose()
        self.header = None
