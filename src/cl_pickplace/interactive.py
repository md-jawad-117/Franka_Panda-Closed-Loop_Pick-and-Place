"""Small, testable policies for interactive input and episode presentation."""
from queue import Empty


def drain_commands(commands):
    """Collapse disturbance key-repeat bursts, respecting reset/mode boundaries."""
    result=[];seen=set()
    while True:
        try:command=commands.get_nowait()
        except Empty:return result
        if command in ('slot','push','nudge'):
            if command in seen:continue
            seen.add(command)
        else:seen.clear()
        result.append(command)


def auto_reset_due(done_wall,now,enabled,delay):
    return bool(enabled and done_wall is not None and now-done_wall>delay)
