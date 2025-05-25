import re
from typing import Protocol, Tuple, TypeAlias

import adbutils
import cv2 as cv
import numpy as np
from cv2.typing import MatLike

Point: TypeAlias = Tuple[int | float, int | float]


class DeviceProto(Protocol):
    def is_screen_on(self) -> bool:
        pass

    def screenshot(self) -> MatLike:
        pass

    def click(self, pos: Point) -> None:
        pass

    def back(self):
        pass

    def current_focus(self) -> str:
        pass

    def swipe(self, start: Point, end: Point, duration: float) -> None:
        pass


class AdbDevice(DeviceProto):
    active_pkg_re = r"mCurrentFocus=Window{[^ ]+ [^ ]+ ([a-zA-A0-9_.]+)/.+}"

    def __init__(self, serial: str | None):
        self.d = adbutils.device(serial=serial)

    def is_screen_on(self) -> bool:
        return self.d.is_screen_on()

    def screenshot(self) -> MatLike:
        screen = self.d.screenshot()
        return cv.cvtColor(np.array(screen), cv.COLOR_RGB2GRAY)

    def click(self, pos: Point):
        self.d.click(x=pos[0], y=pos[1])

    def back(self):
        self.d.keyevent(4)

    def swipe(self, start: Point, end: Point, duration: float) -> None:
        self.d.swipe(sx=start[0], sy=start[1], ex=end[0], ey=end[1], duration=duration)

    def current_focus(self) -> str:
        w = self.d.shell(["dumpsys", "window"], timeout=2)
        m = re.search(self.active_pkg_re, w)
        if m is None:
            raise Exception("No active package found")

        return m.group(1)


class MockDevice(DeviceProto):
    seqn = 0
    seq = []

    def is_screen_on(self) -> bool:
        return True

    def screenshot(self) -> MatLike:
        f = self.seq[self.seqn]
        self.seqn += 1
        return cv.imread(f, cv.IMREAD_GRAYSCALE)

    def click(self, pos: Point):
        print(f"Mock: click at {pos}")

    def back(self):
        print("Mock: back")
