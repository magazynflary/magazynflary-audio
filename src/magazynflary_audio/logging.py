import logging


def setup_logging(level: int | str = logging.INFO) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=level,
    )
