from typing import Annotated

import cv2 as cv
import typer
from rich.console import Console

from pkmnpkt.bot import Bot
from pkmnpkt.device import DeviceProto, MockDevice, AdbDevice

err_console = Console(stderr=True)
app = typer.Typer()
d: DeviceProto
b: Bot


@app.callback()
def hook(
    serial: Annotated[str | None, typer.Option()] = None,
    mock: Annotated[bool, typer.Option()] = False,
) -> None:
    global d, b
    if mock:
        d = MockDevice()
    else:
        d = AdbDevice(serial=serial)

    if not d.is_screen_on():
        err_console.print("screen is off")
        raise typer.Exit(code=1)

    b = Bot(d)


@app.command("extract-collection", help="Extract collection into CSV")
def cmd_extract_collection():
    b.reset_home()


@app.command("send-like", help="Send like to all your friendlist")
def cmd_send_like():
    b.reset_home()
    b.sequence_gift()


@app.command("grab")
def cmd_grab():
    screen = d.screenshot()
    cv.imwrite("screen.png", screen)


if __name__ == "__main__":
    app()
