"""Run one formal arm in the isolated 501--510 seed replication."""

from WorldModel.evaluation.phase_c_s1_hungarian_replication_501_510 import (
    install,
)


install()

from WorldModel.evaluation.run_phase_c_s1_hungarian_arm import main  # noqa: E402


if __name__ == "__main__":
    main()

