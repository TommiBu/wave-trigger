"""Minimal callback example: wave and something happens."""

from wave_trigger import WaveTrigger


def hello_world(event):
    print(f"hello world  ({event})")


if __name__ == "__main__":
    WaveTrigger(on_wave=hello_world).run()
