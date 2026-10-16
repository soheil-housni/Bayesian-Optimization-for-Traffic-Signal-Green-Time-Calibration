import sympy as sp


def get_upper_bounds(initial_tl_df):

    initial_tl_rest_durations=initial_tl_df.loc[initial_tl_df["to_change"]==0].groupby("tl")["initial_duration"].sum().reset_index().sort_values(by=["tl"])
    def finding_upper_bound(initial_duration):
        x = sp.symbols('x', integer=True, positive=True)
        equation=sp.Eq(90-x-initial_duration,1)
        solution = sp.solve(equation, x)
        return solution[0]
    initial_tl_rest_durations["upper_bound"]=initial_tl_rest_durations["initial_duration"].apply(finding_upper_bound)
    upper_bounds=list(initial_tl_rest_durations["upper_bound"].values)

    return upper_bounds