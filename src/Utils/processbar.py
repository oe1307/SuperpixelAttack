def pbar(step: int, total: int, msg: str = "") -> None:
    if step != total:
        print(
            f"\r{msg} ["
            + "#" * int(step / total * 10)
            + " " * int((1 - step / total) * 10)
            + f"] {step}/{total}"
            + " " * 5,
            end="",
        )
    else:
        print(f"\r{msg} [" + "#" * 10 + f"] {step}/{total}")
