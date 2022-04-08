from itertools import chain
from typing import Iterable, Tuple


def partitions(total: int, boxes: int) -> Iterable[Tuple[int, ...]]:
    if boxes == 1:
        return [(total,)]
    return chain(
        *[
            [
                partition + (last_box_pop,)
                for partition in partitions(total - last_box_pop, boxes - 1)
            ]
            for last_box_pop in range(total + 1)
        ]
    )
