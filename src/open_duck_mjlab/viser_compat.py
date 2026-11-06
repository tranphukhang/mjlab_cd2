"""Workaround for the mjlab 1.6 Viser velocity joystick.

``UniformVelocityCommand.create_gui`` builds a "Max <axis>" slider with
``min=0.1`` and ``initial_value=<range max>``. Our tasks use zero lateral and
yaw ranges, so Viser rejects the slider and ``play --viewer viser`` crashes.
Raise each range max to at least 0.1 only while the GUI is being built.
"""

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

_MIN_SLIDER_MAX = 0.1
_original_create_gui = UniformVelocityCommand.create_gui


def _create_gui(self, *args, **kwargs):
  ranges = self.cfg.ranges
  saved = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
  ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = (
    (lo, max(hi, _MIN_SLIDER_MAX)) for lo, hi in saved
  )
  try:
    return _original_create_gui(self, *args, **kwargs)
  finally:
    ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z = saved


UniformVelocityCommand.create_gui = _create_gui
