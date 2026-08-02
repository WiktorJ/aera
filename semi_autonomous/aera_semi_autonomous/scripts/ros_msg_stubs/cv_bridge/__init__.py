"""CvBridge that refuses to convert (see ../README.md).

Only reached with a TrajectoryDataCollector attached, which the headless
measurement tools never do — so raising here surfaces the mistake instead of
silently writing a dataset with stub-encoded images.
"""


class CvBridge:
    def cv2_to_imgmsg(self, *args, **kwargs):
        raise NotImplementedError(
            "ros_msg_stubs.CvBridge cannot encode images — run data collection "
            "inside the ROS container with the real cv_bridge."
        )

    def imgmsg_to_cv2(self, *args, **kwargs):
        raise NotImplementedError(
            "ros_msg_stubs.CvBridge cannot decode images — run data collection "
            "inside the ROS container with the real cv_bridge."
        )
