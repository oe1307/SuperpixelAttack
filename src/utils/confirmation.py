def confirmation(message="Are you sure you want to continue? [y/n]: "):
    while True:
        confirm = input(message)
        if confirm in ["y", "Y", "yes"]:
            break
        elif confirm in ["n", "N", "no", "No"]:
            exit()
