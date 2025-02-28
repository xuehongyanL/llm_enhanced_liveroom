from bisect import bisect_left
from collections import namedtuple

import pandas as pd

Explain = namedtuple('Explain', ['name', 'start', 'end'])


class ExplainChecker:
    def __init__(self, df_explain: pd.DataFrame) -> None:
        self.explains = [Explain(row['product_name'],
                                 row['start_timeline'],
                                 row['end_timeline']
                                 ) for _, row in df_explain.iterrows()]

    # Naive implementation
    # Use binary search version if needed
    def get_explaining(self, ts: pd.Timestamp, default: str = '') -> str:
        # Sometimes explain may <1min
        # e.g. A(10:34~10:45) - B(10:45~10:45) - C(10:45~11:30)
        # We set A (earlist occurrence) as explaining at 10:45
        for ex in self.explains:
            if ex.start <= ts <= ex.end:
                return ex.name
        return default


def get_explaining(timeline, df_explain, default: str = ''):
    return timeline.apply(ExplainChecker(df_explain).get_explaining, default=default)
