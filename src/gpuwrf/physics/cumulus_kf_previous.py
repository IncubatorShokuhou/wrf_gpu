"""Compatibility name for the distinct, source-qualified old-KF cu=99 kernel.

The v0.17 KF-eta candidate alias has been replaced by the actual KFCPS/KFPARA
port. Experimental; GPU/coupled-forecast qualification pending.
"""
from gpuwrf.physics.cumulus_kf99 import step_kf99_column as step_previous_kf_column
__all__ = ["step_previous_kf_column"]
