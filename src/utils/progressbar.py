from .logging import setup_logger

logger = setup_logger(__name__)


class ProgressBar:
    """print progress bar"""

    def __init__(self):
        self.color = {
            "default": "\033[0m",
            "debug": "\033[36m",
            "info": "\033[32m",
            "warning": "\033[33m",
            "error": "\033[31m",
            "critical": "\033[31m",
        }

    def __call__(self, step: int, total: int, f: str = "", b: str = "") -> None:
        color_code = self.color["default"]
        self._base(color_code, step, total, f, b)

    def debug(self, step: int, total: int, f: str = "", b: str = "") -> None:
        color_code = self.color["debug"]
        if logger.root.level < 20:
            self._base(color_code, step, total, f, b)

    def info(self, step: int, total: int, f: str = "", b: str = "") -> None:
        color_code = self.color["info"]
        if logger.root.level < 30:
            self._base(color_code, step, total, f, b)

    def warning(self, step: int, total: int, f: str = "", b: str = "") -> None:
        color_code = self.color["warning"]
        if logger.root.level < 40:
            self._base(color_code, step, total, f, b)

    def error(self, step: int, total: int, f: str = "", b: str = "") -> None:
        color_code = self.color["error"]
        if logger.root.level < 50:
            self._base(color_code, step, total, f, b)

    def critical(self, step: int, total: int, f: str = "", b: str = "") -> None:
        color_code = self.color["critical"]
        self._base(color_code, step, total, f, b)

    def _base(self, color_code: str, step: int, total: int, f: str, b: str) -> None:
        if step < total:
            print(
                f"\r{color_code}{f} ["
                + "#" * int(step / total * 10)
                + " " * int((1 - step / total) * 10)
                + f"] {step}/{total} {b}\033[0m     ",
                end="",
            )
        else:
            print(f"\r{color_code}{f} [##########] {step}/{total} {b}\033[0m")


pbar = ProgressBar()
