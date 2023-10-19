prompt = {
    "white": "\033[37m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def printc(color: str, *args, **kwargs):
    print(prompt[color], *args, "\033[0m", **kwargs)


class ProgressBar:
    def __init__(
        self, total: int, fmsg: str = "", bmsg: str = "", color="white", start=0
    ):
        """
        Args:
            total (int): total number of iterations
            fmsg (str): front message. Defaults to "".
            bmsg (str): back message. Defaults to "".
            color (str): color of the progress bar. Defaults to "white".
            start (int): start number. Defaults to 0.
        """
        self.total = total
        self.iter = start
        self.fmsg = fmsg
        self.bmsg = bmsg
        self.color = prompt[color]
        bar = " [          ] "
        print(
            f"{self.color}{self.fmsg}{bar}{start}/{self.total} {self.bmsg}\033[0m",
            end="\r",
        )

    def step(self):
        self.iter += 1
        assert self.iter <= self.total
        percent = int((self.iter) / self.total * 10)
        bar = " [" + ("#" * percent).ljust(10, " ") + "] "
        print(
            f"{self.color}{self.fmsg}{bar}{self.iter}/{self.total} {self.bmsg}\033[0m",
            end="\r",
        )

    def update(self, step):
        self.iter = step
        assert self.iter <= self.total
        percent = int((self.iter) / self.total * 10)
        bar = " [" + ("#" * percent).ljust(10, " ") + "] "
        print(
            f"{self.color}{self.fmsg}{bar}{self.iter}/{self.total} {self.bmsg}\033[0m",
            end="\r",
        )

    def end(self):
        bar = " [##########] "
        print(
            f"{self.color}{self.fmsg}{bar}{self.iter}/{self.total} {self.bmsg}\033[0m"
        )
