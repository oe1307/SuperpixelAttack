def pbar(step: int, total: int, front_msg: str = "", back_msg: str = "") -> None:
    """print progress bar"""
    if step != total:
        print(
            f"\r{front_msg} ["
            + "#" * int(step / total * 10)
            + " " * int((1 - step / total) * 10)
            + f"] {step}/{total}"
            + " " * 5
            + back_msg,
            end="",
        )
    else:
        print(f"\r{front_msg} [" + "#" * 10 + f"] {step}/{total} {back_msg}")
