"""Generator example: you own the loop, so the action can be anything."""

import time

from wave_trigger import WaveTrigger

if __name__ == "__main__":
    with WaveTrigger() as trigger:
        for i, event in enumerate(trigger.events(), 1):
            print(f"#{i}  {event}  at {time.strftime('%H:%M:%S')}")
            if i == 5:
                break
