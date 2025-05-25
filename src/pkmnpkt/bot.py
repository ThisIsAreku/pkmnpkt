import time
from typing import Dict, Generator, NamedTuple, Tuple, TypeAlias

import cv2 as cv
import numpy as np
import pytesseract
from cv2.typing import MatLike

from pkmnpkt.assets import Asset, load_asset
from pkmnpkt.device import DeviceProto

lvl_config = "--psm 8 --dpi 72 --oem 1 -c tessedit_do_invert=0 -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_char_whitelist=0123456789"
pseudo_config = "--psm 8 --dpi 72 --oem 1 -c tessedit_do_invert=0 -c load_system_dawg=0 -c load_freq_dawg=0 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


class Box(NamedTuple):
    left: int
    top: int
    width: int
    height: int

    def center(self) -> Tuple[int, int]:
        return int(self.left + self.width / 2), int(self.top + self.height / 2)

    def bottom_left(self) -> Tuple[int, int]:
        return int(self.left), int(self.top + self.height)


class UserInfo(NamedTuple):
    name: str
    level: int
    coords: Box


UserInfoDict: TypeAlias = Dict[str, UserInfo]


def prepare_roi(img_cropped: MatLike) -> MatLike:
    otsu = cv.threshold(img_cropped, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    kernel = np.ones((2, 1), dtype=np.uint8)
    erosion = cv.erode(otsu, kernel, iterations=1)
    kernel = np.ones((1, 2), dtype=np.uint8)
    erosion = erosion + cv.erode(otsu, kernel, iterations=1)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilated = cv.dilate(erosion, kernel, iterations=1)
    mask = dilated / 255
    img_cropped = otsu * mask

    # converts back to numpy array (uint8) again and applies a small blur to blend edges
    img_cropped = np.uint8(img_cropped)
    img_cropped = cv.blur(img_cropped, (2, 2))

    return img_cropped


def deduplicate_matches(
    matches: Tuple[np.ndarray, np.ndarray], min_dist: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    filtered = []
    for x, y in zip(matches[1], matches[0]):
        point = (x, y)
        if all(
            np.linalg.norm(np.array(point) - np.array(f)) > min_dist for f in filtered
        ):
            filtered.append(point)
    if not filtered:
        return np.array([], dtype=int), np.array([], dtype=int)
    xs, ys = zip(*filtered)
    return np.array(ys, dtype=int), np.array(xs, dtype=int)


def locate(img: MatLike, template: Asset):
    template = load_asset(template)

    needleHeight, needleWidth = template.shape[:2]

    result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)

    return result, needleHeight, needleWidth


def locate_best(img: MatLike, template: Asset, confidence: float = 0.998) -> Box | None:
    result, needleHeight, needleWidth = locate(img, template)
    _, max_val, _, max_loc = cv.minMaxLoc(result)

    if max_val < confidence:
        return None

    return Box(max_loc[0], max_loc[1], needleWidth, needleHeight)


def locate_all(
    img: MatLike, template: Asset, limit: int = 10, confidence: float = 0.998
) -> Generator[Box]:
    step = 1
    region = (0, 0)

    result, needleHeight, needleWidth = locate(img, template)

    match_indices = np.arange(result.size)[(result > confidence).flatten()]
    matches = np.unravel_index(match_indices[:limit], result.shape)

    unique_points = deduplicate_matches(matches, min_dist=needleWidth // 2)

    if len(unique_points[0]) == 0:
        raise Exception(
            "Could not locate the image (highest confidence = %.3f)" % result.max()
        )

    # use a generator for API consistency:
    matchx = unique_points[1] * step + region[0]  # vectorized
    matchy = unique_points[0] * step + region[1]
    for x, y in zip(matchx, matchy):
        yield Box(x, y, needleWidth, needleHeight)


class Bot:
    def __init__(self, device: DeviceProto):
        self.device = device

    def _focus_bound_guard(self):
        if self.device.current_focus() != "jp.pokemon.pokemontcgp":
            raise Exception("out of bounds")

    def _back(self):
        """
        Safe version of device back.
        Check if we are still within boundaries after going back
        """
        self.device.back()
        self._focus_bound_guard()

    def _friendlist_scroll_down(self):
        """
        A more controlled scrolling version
        """
        self.device.swipe(start=(0.5, 0.80), end=(0.5, 0.30), duration=2.5)
        self.device.click((0, 0.5))  # click on the center-left pixel to stop scrolling

    def _userdetail_scroll_down(self):
        """
        A more controlled scrolling version, tuned for userdetail screen
        """
        self.device.swipe(start=(0.5, 0.80), end=(0.5, 0.50), duration=0.5)
        self.device.click((0, 0.5))  # click on the center-left pixel to stop scrolling

    def _wait_on(
        self, asset: Asset, attempts: int = 5, interval: float = 0.250
    ) -> Box | None:
        for n in range(attempts):
            screen = self.device.screenshot()
            target = locate_best(screen, asset)
            if target is not None:
                return target

            time.sleep(interval)

        return None

    def reset_home(self):
        self._focus_bound_guard()
        screen = self.device.screenshot()  # screen is reused if we are already on main
        for n in range(10):
            marker_on_main = locate_best(screen, Asset.MARKER_ON_MAIN)
            if marker_on_main is not None:
                break

            self._back()
            time.sleep(1)
            screen = self.device.screenshot()

        goto_home_btn = locate_best(screen, Asset.BUTTON_GOTO_HOME_ACTIVE)
        if goto_home_btn is not None:
            return  # we are ready

        goto_home_btn = locate_best(screen, Asset.BUTTON_GOTO_HOME)
        if goto_home_btn is None:
            raise Exception("Could not locate the home button")

        self.device.click(goto_home_btn.center())
        time.sleep(4)  # loading home is slow

    def sequence_gift(self):
        self._focus_bound_guard()
        screen = self.device.screenshot()
        goto_community_btn = locate_best(screen, Asset.BUTTON_GOTO_COMMUNITY)
        if goto_community_btn:
            self.device.click(goto_community_btn.center())

        time.sleep(2.5)
        open_friendlist_btn = self._wait_on(Asset.BUTTON_OPEN_FRIENDLIST, interval=0.5)
        if not open_friendlist_btn:
            raise Exception("Could not locate the open friendlist button")

        self.device.click(open_friendlist_btn.center())

        self._subsequence_process_friend_rows()

    def _subsequence_process_friend_rows(self):
        visited_users: UserInfoDict = {}

        def _process_single_user(user: UserInfo):
            self.device.click(user.coords.bottom_left())
            time.sleep(0.8)
            if self._wait_on(Asset.MARKER_ON_USERDETAIL) is None:
                raise Exception("sequence break")
            self._userdetail_scroll_down()
            screen = self.device.screenshot()
            b = locate_best(
                screen, Asset.BUTTON_OPEN_SHOWCASE, confidence=0.97
            )  # button is hard to target...
            if b is None:
                self._back()
                return

            self.device.click(b.center())
            time.sleep(1.5)
            self.device.click((0.5, 0.306))  # like button is always in the same place
            time.sleep(0.5)
            self._back()
            time.sleep(1)
            self._back()
            time.sleep(0.5)
            if self._wait_on(Asset.MARKER_ON_FRIENDLIST) is None:
                raise Exception("sequence break")

        def _process_screen() -> UserInfoDict:
            result: UserInfoDict = {}
            screen = self.device.screenshot()
            for b in locate_all(
                screen, Asset.MARKER_FRIEND_CELL_EDGE, limit=16, confidence=0.95
            ):
                lvl_roi = screen[
                    b.top + 75 : b.top + 75 + 50, b.left - 560 : b.left - 560 + 100
                ]
                pseudo_roi = screen[
                    b.top + 130 : b.top + 130 + 50, b.left - 600 : b.left - 600 + 300
                ]
                lvl_roi = prepare_roi(lvl_roi)
                pseudo_roi = prepare_roi(pseudo_roi)

                lvl = (
                    pytesseract.image_to_string(lvl_roi, config=lvl_config)
                    .replace("\f", "")
                    .strip()
                )
                pseudo = (
                    pytesseract.image_to_string(pseudo_roi, config=pseudo_config)
                    .replace("\f", "")
                    .strip()
                )

                userinfo = UserInfo(pseudo, lvl, b)
                if userinfo.name in visited_users:
                    continue

                if userinfo.name in result:
                    continue

                result[userinfo.name] = userinfo

                print(f"processing {userinfo.name}")
                _process_single_user(userinfo)

            return result

        if self._wait_on(Asset.MARKER_ON_FRIENDLIST) is None:
            raise Exception("sequence break")

        for n in range(10):
            new_visited_users = _process_screen()
            if len(new_visited_users) == 0:
                break

            print(f"round visited users: {new_visited_users}")
            visited_users.update(new_visited_users)
            self._friendlist_scroll_down()

        print("Done")
        print(f"All visited users: {visited_users}")
