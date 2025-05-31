def write_progress(current: int, max: int, title: str = "Progress"):
    """Writes progress to stdout and updates interactively.

    Write progress to stdout and updates interactively,
    it is important that nothing else writes to stdout during updates.
    """
    print(f"\r{title}: {current}/{max}", end="")
