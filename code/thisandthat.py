import numpy as np
import pandas as pd
tmp = pd.date_range("2018-01-01","2018-01-02", freq = "min")
df_tmp = pd.DataFrame({"dep" : np.random.randint(1,10,len(tmp))},
                      index = tmp).sample(frac = 0.5).sort_index()
df_tmp["roll_backward_inclusive"] = df_tmp.dep.resample("min").asfreq().rolling("3min").mean()
df_tmp["roll_forward_inclusive"] = df_tmp.dep.resample("min").asfreq().shift(-2,"min").rolling("3min").mean()
df_tmp["roll_forward_exclusive"] = df_tmp.dep.resample("min").asfreq().shift(-3,"min").rolling("3min").mean()



df_orig = (pd.DataFrame({"A": np.random.randint(0, 10, 90),
                      "B": np.random.randint(0, 10, 90),
                      "day": pd.date_range('2018-01-01', '2018-03-31')})
        .melt("day", ["A","B"], var_name = "g")
        .set_index("day"))
df_orig.drop(df_orig.index[21], inplace = True)
df_orig.dtypes
df_ts = df_orig.reset_index().pivot("day","g","value")

df_mon = df_ts.resample("W-MON").asfreq().shift(1).rolling(3, min_periods = 1).mean()

df_all = df_ts.copy().resample("D").asfreq()
df_all["dayofweek"] = df_all.index.weekday
df_days= df_all.shift(7).groupby("dayofweek", group_keys = False).rolling(3, min_periods = 1).mean().drop("dayofweek", axis = 1)
tmp = df_days.reset_index().melt("day", var_name = "g").set_index("day")

