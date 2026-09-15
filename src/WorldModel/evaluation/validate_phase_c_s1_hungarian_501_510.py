"""Validate the isolated 501--510 seed replication."""

from WorldModel.evaluation.phase_c_s1_hungarian_replication_501_510 import (
    install,
)


install()

from WorldModel.evaluation.validate_phase_c_s1_hungarian_491_500 import (  # noqa: E402
    main,
)


if __name__ == "__main__":
    main()

