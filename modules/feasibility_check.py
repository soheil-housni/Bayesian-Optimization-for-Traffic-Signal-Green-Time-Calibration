import numpy as np
import pandas as pd
from .green_times_function import update_green_time_df


def check_feasibility(g_vec,initial_tl_df):

    tl_df_updated,g_vec_all=update_green_time_df(g_vec,initial_tl_df)
    print(g_vec_all)
    return np.all(g_vec_all > 0)
    