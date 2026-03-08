"""
GPIO button abstraction — BCM GPIO17 (active-low, internal pull-up).

On Raspberry Pi / embedded targets:
  Pressing the physical button calls callback() in a background thread.

On Windows / x86 VM (RPi.GPIO unavailable):
  GPIO is stubbed out; call simulate_press() or use the F1 key shortcut
  in the GUI to test button-triggered actions.
"""

import threading

try:
    import RPi.GPIO as GPIO  # type: ignore
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

BUTTON_PIN = 17  # BCM GPIO17


class GPIOHandler:
    """Monitor GPIO17 for button presses and invoke a callback."""

    def __init__(self, callback=None):
        self._callback = callback
        if _GPIO_AVAILABLE:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
            GPIO.add_event_detect(
                BUTTON_PIN,
                GPIO.FALLING,
                callback=self._on_press,
                bouncetime=300,
            )

    # ---- internal ---------------------------------------------------

    def _on_press(self, channel):  # called from RPi.GPIO's event thread
        if self._callback:
            self._callback()

    # ---- public -----------------------------------------------------

    def simulate_press(self):
        """Trigger the callback in a background thread (desktop testing)."""
        if self._callback:
            threading.Thread(target=self._callback, daemon=True).start()

    def cleanup(self):
        if _GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return _GPIO_AVAILABLE
