def pbar(msg, step, total):
    if step != total:
        print(
            f"\r{msg} ["
            + "#" * int(step / total * 10)
            + " " * int((1 - step / total) * 10)
            + f"] {step}/{total}",
            end="",
        )
    else:
        print(f"\r{msg} [" + "#" * 10 + f"] {step}/{total}")
