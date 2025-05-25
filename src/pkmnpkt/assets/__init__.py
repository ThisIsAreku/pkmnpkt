import importlib.resources as pkg_resources
from enum import Enum

import cv2 as cv
import numpy as np


class Asset(Enum):
    BUTTON_OPEN_FRIENDLIST = "btn_open_friendlist.png"

    BUTTON_GOTO_HOME = "btn_goto_home.png"
    BUTTON_GOTO_HOME_ACTIVE = "btn_goto_home__active.png"
    BUTTON_GOTO_COMMUNITY = "btn_goto_community.png"

    BUTTON_OPEN_SHOWCASE = "btn_open_showcase.png"

    MARKER_ON_MAIN = "marker_on_main.png"
    MARKER_ON_USERDETAIL = "marker_on_user_detail.png"
    MARKER_ON_FRIENDLIST = "marker_on_friendlist.png"
    MARKER_FRIEND_CELL_EDGE = "marker_friend_cell_edge.png"


def load_asset(asset: Asset):
    with pkg_resources.files(__package__).joinpath(asset.value).open("rb") as f:
        img_array = np.asarray(bytearray(f.read()), dtype=np.uint8)
        return cv.imdecode(img_array, cv.IMREAD_GRAYSCALE)
