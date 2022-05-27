
# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from datetime import datetime
import gc

# Specific parameters
n_sample = None
n_jobs = 4
ids = ["id"]
plt.ioff(); matplotlib.use('Agg')
# plt.ion(); matplotlib.use('TkAgg')
d_comb = {1: ["dummy"],
          2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
          6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
          10: ["item_id"], 11: ["item_id", "state_id"], 12: ["item_id", "store_id"]} # Aggregation levels
suffix = "" if n_sample is None else "_" + str(n_sample)
df_help = pd.read_feather("df_help" + suffix + ".ftr")

# Read data
df_submit = (pd.melt(pd.read_csv(dataloc + "submit.csv").iloc[0:30490]
                     .rename(columns = {"F" + str(x): "d_" + str(x + 1913) for x in range(1, 29)}),
                     id_vars = "id", var_name = "date", value_name = "yhat")
             .assign(id = lambda x: x["id"].str.rsplit("_", 1).str[0]))
df_submit = df_submit.query("yhat != 0")
'''
df_submit = (df_test
             .assign(id = lambda x: x["id"].str.rsplit("_", 1).str[0])
             .assign(date = lambda x: "d_" + ((x["date"] - x["date"].min()).dt.days + 1914).astype("str"))
             [["id","date","yhat"]])
'''
df_truth = (pd.melt(pd.read_csv(dataloc + "sales_train_evaluation.csv",
                                usecols = list(["id"]) + ["d_" + str(x) for x in range(1914, 1942)]),
                    id_vars = "id", var_name = "date", value_name = "demand")
            .assign(id = lambda x: x["id"].str.rsplit("_", 1).str[0]))
df_ids = (pd.read_csv(dataloc + "sales_train_validation.csv", usecols = range(0, 6))
          .assign(id = lambda x: x["id"].str.rsplit("_", 1).str[0]))
df_holdout = df_submit.merge(df_truth, how = "left").merge(df_ids, how = "left", on = "id")

# Rmse
print(rmse(df_holdout["yhat"], df_holdout["demand"]))
print(df_holdout["yhat"].mean(), df_holdout["demand"].mean())
print(df_holdout["demand"].mean()/df_holdout["yhat"].mean())

df_rmse = pd.DataFrame()
for key in d_comb:
    #key = 12
    df_tmp = (df_holdout.assign(dummy = "dummy")
              .groupby(d_comb[key] + ["date"])["demand", "yhat"].sum().reset_index("date", drop = True)
              .groupby(d_comb[key]).apply(lambda x: pd.Series({"rmse": rmse(x["demand"], x["yhat"])}))
              .assign(key = key)
              .reset_index())
    df_rmse = pd.concat([df_rmse, df_tmp], ignore_index = True)
df_tmp = df_rmse.merge(df_help, how = "left").eval("wrmsse = sales * rmse/rmse_denom")
df_tmp.groupby("key")["wrmsse"].sum()
print(df_tmp["wrmsse"].sum())


