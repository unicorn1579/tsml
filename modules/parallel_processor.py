from itertools import chain
from typing import List
from joblib import Parallel, delayed
from modules.common import TimeSeries
import modules.data_reader as dr


def get_time_series(params: dr.InputParams) -> List[TimeSeries]:
    time_series: List[List[TimeSeries]] = Parallel(n_jobs=-1, backend='threading', verbose=10)(
        delayed(dr.get_time_series)(params=params, ticket_name=ticket_name)
        for ticket_name in params.get_tickets_names()
    )
    return list(chain.from_iterable(time_series))

