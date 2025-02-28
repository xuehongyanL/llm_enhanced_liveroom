import math
from collections import namedtuple
from typing import Tuple

import pandas as pd
from chinese_calendar import is_workday as check_work_day


def periodic_encode(series, period) -> Tuple[pd.Series, pd.Series]:
    return (2 * math.pi / period * series).apply(math.cos), \
           (2 * math.pi / period * series).apply(math.sin)


DTFeatures = namedtuple('DTFeatures', ['dow_cos',
                                       'dow_sin',
                                       'hour_cos',
                                       'hour_sin',
                                       'is_workday',
                                       'is_holiday'])


def get_dt_features(raw_series) -> DTFeatures:
    dt_series = pd.to_datetime(raw_series)
    dow = dt_series.dt.day_of_week
    dow_cos, dow_sin = periodic_encode(dow, 7)
    hour = dt_series.dt.hour
    hour_cos, hour_sin = periodic_encode(hour, 24)
    is_workday = dt_series.apply(check_work_day).astype(float)
    is_holiday = 1. - is_workday

    return DTFeatures(dow_cos=dow_cos,
                      dow_sin=dow_sin,
                      hour_cos=hour_cos,
                      hour_sin=hour_sin,
                      is_workday=is_workday,
                      is_holiday=is_holiday)
