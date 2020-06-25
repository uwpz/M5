
# Important columns
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from datetime import datetime
import gc

# Specific parameters
n_sample = 1000
n_jobs = 16
plt.ioff(); matplotlib.use('Agg')
# plt.ion(); matplotlib.use('TkAgg')

begin = datetime.now()


# ######################################################################################################################
#  ETL
# ######################################################################################################################

# --- Read data and 1st FE ---------------------------------------------------------------------------------------------

# Sales, calendar, prices
if n_sample is None:
    #df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv")
    df_sales_orig = pd.read_csv(dataloc + "sales_train_evaluation.csv") # TODO
else:
    #df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv").sample(n = int(n_sample), random_state = 1)
    df_sales_orig = pd.read_csv(dataloc + "sales_train_evaluation.csv").sample(n = int(n_sample), random_state = 1) # TODO
#df_sales_orig[["d_" + str(x) for x in range(1914, 1942)]] = pd.DataFrame([[np.nan for x in range(1914, 1942)]]) # TODO
df_sales_orig[["d_" + str(x) for x in range(1942, 1970)]] = pd.DataFrame([[np.nan for x in range(1942, 1970)]]) # TODO
df_sales = pd.melt(df_sales_orig, id_vars = df_sales_orig.columns.values[:6],
                   var_name = "d", value_name = "demand")
holidays = ["ValentinesDay","StPatricksDay","Easter","Mother's day","Father's day","IndependenceDay",
            "Halloween","Thanksgiving","Christmas", "NewYear"]
df_calendar = (pd.read_csv(dataloc + "calendar.csv", parse_dates=["date"])
               .assign(dummy = 1)
               .assign(event_name = lambda x: np.where(x["event_name_2"].isin(["Easter", "Cinco De Mayo"]),
                                                       x["event_name_2"], x["event_name_1"]))
               .assign(event_type = lambda x: np.where(x["event_name_2"].isin(["Easter", "Cinco De Mayo"]),
                                                       x["event_type_2"], x["event_type_1"]))
               .assign(event = lambda x: x["event_name"].notna().astype("int"))
               .assign(next_event = lambda x: x["event_name"].fillna(method = "bfill"))
               .assign(prev_event = lambda x: x["event_name"].fillna(method = "ffill"))
               .assign(prev_event = lambda x: x["prev_event"].fillna("Unknown"))
               .assign(days_before_event = lambda x: (x.groupby(["year", "next_event"])["dummy"]
                                                      .transform(lambda y: y.sum() - y.cumsum())))
               .assign(days_after_event = lambda x: (x.groupby(["year", "prev_event"])["dummy"]
                                                     .transform(lambda y: y.cumsum() - 1)))
               .assign(holiday_name = lambda x: np.where(x["event_name_1"].isin(holidays), x["event_name"], np.nan))
               .assign(holiday_name = lambda x: np.where(x["event_name_2"].isin(holidays),
                                                         x["event_name_2"], x["holiday_name"]))
               .assign(holiday = lambda x: x["holiday_name"].notna().astype("int"))
               .assign(next_holiday = lambda x: x["holiday_name"].fillna(method = "bfill"))
               .assign(prev_holiday = lambda x: x["holiday_name"].fillna(method = "ffill"))
               .assign(prev_holiday = lambda x: x["prev_holiday"].fillna("Unknown"))
               .assign(days_before_holiday = lambda x: (x.groupby(["year", "next_holiday"])["dummy"]
                                                        .transform(lambda y: y.sum() - y.cumsum())))
               .assign(days_after_holiday = lambda x: (x.groupby(["year", "prev_holiday"])["dummy"]
                                                       .transform(lambda y: y.cumsum() - 1)))
               .drop(columns = "dummy")
               )
df_prices = (pd.read_csv(dataloc + "sell_prices.csv")
             .assign(sell_price_avg = lambda x: (x.reset_index(drop = False)
                                                 .groupby(["store_id", "item_id"])
                                                 .apply(lambda y: (y.set_index("index")
                                                                   ["sell_price"].rolling(8, min_periods = 1).mean()))
                                                 .reset_index(["store_id", "item_id"], drop = True)))
             .assign(price_ratio_to_8week_rolling = lambda x: x["sell_price"] / x["sell_price_avg"])
             .drop(columns = ["sell_price_avg"]))

# Merge all
df = (df_sales
      .merge(df_calendar.drop(columns = ["weekday", "wday", "month", "year",
                                         "event_name_1", "event_type_1", "event_name_2", "event_type_2"]),
             how = "left", on = "d")
      .merge(df_prices, how = "left", on = ["store_id", "item_id", "wm_yr_wk"]))

df["anydemand"] = np.where(df["demand"] > 0, 1, np.where(df["demand"].notna(), 0, np.nan))
df["sell_price_isna"] = np.where(df["sell_price"].isna(), 1, 0)  # no sales if ==1
df["sales"] = (df["demand"] * df["sell_price"]).fillna(0)
df["snap"] = np.where(df["state_id"] == "CA", df["snap_CA"],
                      np.where(df["state_id"] == "TX", df["snap_TX"], df["snap_WI"]))  # compress snap
df = df.drop(columns = ["snap_CA", "snap_TX", "snap_WI"])
df["id_copy"] = df["id"].str.rsplit('_',1).str[0]
df = df.merge((df.query("anydemand == 1").groupby("id")[["date"]].min()
               .rename(columns = {"date": "min_date"}).reset_index()),
              how = "left")

'''
# --- Some checks -----------------------------------------------------------------------------------------------------

# Basic
df_prices.describe(include="all")
df_values = create_values_df(df_sales, 10)
df_sales_orig.describe(include="all")

# "real missings"
sum(df["demand"] == 0)/len(df)
df_tmp = df.loc[df["sell_price"].notna()]; sum(df_tmp["demand"] == 0)/len(df_tmp)

# Demand per id -> no scaling needed
df["demand"].loc[df["demand"]>0].value_counts().plot()
np.quantile(df["demand"].loc[df["demand"]>0], q = np.arange(0,1,0.01))
plt.hist(np.log(df["demand"].loc[df["demand"]>0]), bins=50)
df[["id","cat_id","demand"]].loc[df["demand"] > 0].groupby(["cat_id","id"])["demand"].median().unstack("cat_id").boxplot()
plt.hist(df["demand"].loc[df["demand"]>0], density=True, cumulative=True, label='CDF',
         histtype='step', alpha=0.8, color='k', range = (0,10))
def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)
fig, ax = plt.subplots(1, 1)
ax = cdf(df["demand"].loc[df["demand"]>0])

# Demand per store: any "closings" or trends?
df.loc[df["sell_price"].notna()].groupby(["date","store_id"])["demand"].mean().unstack().plot()
(df.loc[df["sell_price"].notna()].groupby(["date","store_id"])["demand"].mean().unstack()
 .rolling("60D", min_periods=60).mean().plot())

# Missing pattern
df_tmp["item_id"].value_counts()
fig, ax = plt.subplots(1,1)
sns.heatmap(df_tmp.assign(anydemand = lambda x: (x["demand"] > 0).astype("int"))
                  .groupby(["date", "id"])["anydemand"].mean().unstack("date"))
fig.tight_layout()
(df.assign(sell_price_isna = lambda x: x["sell_price"].isna())
 .groupby(["date", "cat_id"])["sell_price_isna"].mean().unstack("cat_id").rolling("90D", min_periods=90).mean().plot())
(df.loc[df["sell_price"].notna()].assign(anydemand = lambda x: x["demand"]>0)
 .groupby(["date", "cat_id"])["anydemand"].mean().unstack("cat_id").rolling("90D", min_periods=90).mean().plot())
df_tmp = (df.merge(df.loc[df["demand"] > 0].groupby("id")["date"].agg(min_date = ("date", "min")).reset_index(),
                   how = "left")
          .query("date > min_date"))
(df_tmp.loc[df_tmp["sell_price"].notna()].assign(anydemand = lambda x: x["demand"]>0)
 .groupby(["date", "cat_id"])["anydemand"].mean().unstack("cat_id").rolling("90D", min_periods=90).mean().plot())

# Promotion check
df_prices["item_id"].value_counts()
sns.lineplot(x="wm_yr_wk", y="sell_price", hue="store_id", 
             data=df_prices.loc[df_prices.item_id == "HOUSEHOLD_1_309"])


# --- Time series plots -----------------------------------------------------------------------------------------

ts = df.query("sell_price_isna==0").groupby("date")["demand"].mean()
#ts = df.query("sell_price_isna==0 and cat_id == 'FOODS'").groupby("date")["anydemand"].mean()
#ts = df.groupby(["date","cat_id"])["demand"].mean().unstack()

# STL decomposition
sdec1 = seasonal_decompose(ts, freq=7)
sdec1.plot()
sdec2 = seasonal_decompose(ts - sdec1.seasonal, freq=61)
sdec2.plot()

# Pacf plot
plot_pacf(ts, lags=40)
plot_pacf(ts - sdec1.seasonal, lags=40)
plot_pacf(ts - sdec1.seasonal - sdec2.seasonal, lags = 40)
plot_acf(ts, lags = 40)

# More checks
(df.query("sell_price_isna==0").assign(dayofweek = lambda x: x["date"].dt.dayofweek)
 .groupby(["snap","dayofweek"])["demand"].mean().unstack("snap"))
'''


# --- Transform 1 -----------------------------------------------------------------------------------------------------

del df_sales_orig, df_sales, df_calendar, df_prices
gc.collect()
df = reduce_mem_usage(df)

# Set fold
#df["fold"] = np.where(df["date"] >= "2016-04-25", "test", "train")
df["fold"] = np.where(df["date"] >= "2016-05-23", "test", "train")  # TODO
#df.groupby("fold")["date"].nunique()
#df["myfold"] = np.where(df["date"] >= "2016-04-25", None, np.where(df["date"] >= "2016-03-28", "test", "train"))
df["myfold"] = np.where(df["date"] >= "2016-05-23", None, np.where(df["date"] >= "2016-04-25", "test", "train"))  # TODO
#df.groupby("myfold")["date"].nunique()
df.myfold.describe()

# Add sales weight
df = df.merge((df.query("myfold == 'test'").groupby("id")[["sales"]].sum()
               .rename(columns = {"sales": "weight_sales"}))
               .reset_index(),
              how = "left")

# Add rmse weight
df = df.merge(df[["id", "date", "demand"]].set_index("date").shift(1, "D").rename(columns = {"demand": "lagdemand"})
              .reset_index(),
              how = "left") # Add lagdemand

df = df.merge(df.query("fold == 'train' and date >= min_date")  # & date > min_date
              #.groupby("id").apply(lambda x: x["demand"].mean() / rmse(x["demand"], x["lagdemand"]))
              .groupby("id").apply(lambda x: 1 / rmse(x["demand"], x["lagdemand"]))
              .reset_index(drop = False)
              .rename(columns = {0: "weight_rmse"}),
              how = "left", on = "id")
df["weight_all"] = (df["weight_sales"]) * df["weight_rmse"]
#df[["weight", "weight_rmse", "weight_all"]] = df[["weight", "weight_rmse", "weight_all"]].apply(lambda x: x/x.max())


# --- Eval metric help dataframes: Must be done before setting some demands to na --------------------------------------

df = reduce_mem_usage(df)

# Aggregation levels
d_comb = {1: ["dummy"],
          2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
          6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
          10: ["item_id"], 11: ["item_id", "state_id"], 12: ["item_id", "store_id"]}

# Sales
df_sales_weight = pd.DataFrame()
denom = 12 * df.query("myfold == 'test'")["sales"].sum()
for key in d_comb:
    df_tmp = (df.query("myfold == 'test'").assign(dummy = "dummy").groupby(d_comb[key])[["sales"]].sum()
              .assign(sales = lambda x: x["sales"]/denom)
              .assign(key = key)
              .reset_index())
    df_sales_weight = pd.concat([df_sales_weight, df_tmp], ignore_index = True)

# rmse_denom
df_rmse_denom = pd.DataFrame()
for key in d_comb:
    df_tmp = (df.query("fold == 'train' and date >= min_date").assign(dummy = "dummy")
              .groupby(d_comb[key] + ["date"])["demand", "lagdemand"].sum().reset_index("date", drop = True)
              .groupby(d_comb[key]).apply(lambda x: pd.Series({"rmse_denom": rmse(x["demand"], x["lagdemand"])}))
              .assign(key = key)
              .reset_index())
    df_rmse_denom = pd.concat([df_rmse_denom, df_tmp], ignore_index = True)

# Merge
df_help = df_rmse_denom.merge(df_sales_weight, how = "left")


# --- Transform 2 ------------------------------------------------------------------------------------------------

# Adapt demand due to missing sell_price and xmas outlier
df.loc[df["sell_price_isna"] == 1, ["demand", "anydemand"]] = np.nan
df.loc[df["holiday_name"] == "Christmas", ["demand", "anydemand"]] = np.nan

# Add demand statistics for scaling
df = df.merge(df.groupby("id")["demand"]
              .agg([("demand_mean", "mean"), ("demand_sd", "std"),
                    ("demand_median", "median"), ("demand_upperlimit", lambda x: x.quantile(0.99)),
                    ("demand_iqr", lambda x: x.quantile(0.75) - x.quantile(0.25))])
              .reset_index(),
              how = "left")
      # .assign(demand = lambda x: np.where(x["demand"].isna(), np.nan,
      #                                     np.where(x["myfold"] == "train",
      #                                              x[["demand", "demand_upperlimit"]].min(axis = 1),
      #                                              x["demand"])))


# ######################################################################################################################
#  Time series based FE
# ######################################################################################################################

# --- Basic --------------------------------------------------------------------------------------------------------

df["dayofweek"] = df["date"].dt.dayofweek
df["weekend"] = np.where(df.dayofweek.isin([5, 6]), 1, 0)
df["dayofmonth"] = df["date"].dt.day
df["dayofyear"] = df["date"].dt.dayofyear
df["week"] = df["date"].dt.week
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df = reduce_mem_usage(df)


# --- Advanced ----------------------------------------------------------------------------------------------------

def run_in_parallel(df):

    # Set index
    df = df.set_index(["date"])

    # Lag values (might depend on horizon)
    def demand_lag_calc(shift, columns = ["demand", "snap"]):
        return (df[["id"] + columns]
                .shift(shift, "D")
                .rename(columns = {x: x + "_lag" + str(shift) for x in columns})
                .set_index(["id"], append = True))
    demand_lag = (df[["id"]].set_index(["id"], append = True)
                  .join(demand_lag_calc(shift = 0), how = "left")
                  .join(demand_lag_calc(shift = 1), how = "left")
                  .join(demand_lag_calc(shift = 2), how = "left")
                  .join(demand_lag_calc(shift = 3), how = "left")
                  .join(demand_lag_calc(shift = 4), how = "left")
                  .join(demand_lag_calc(shift = 5), how = "left")
                  .join(demand_lag_calc(shift = 6), how = "left")
                  )
    #df_tmp = demand_lag.reset_index().sort_values(by = ["id"] + ["date"])  # Check

    # Lag same weekday
    # demand_lag_sameweekday = (df[["id"] + ["demand","snap"]]
    #                           .shift(((horizon - 1) // 7 + 1) * 7, "D")
    #                           .rename(columns = {"demand": "demand_lag_sameweekday", "snap": "snap_lag_sameweekday"})
    #                           .set_index(["id"], append = True))

    # Rolling average
    def demand_calc(weekrange, columns = ["demand", "snap"]):
        df_tmp = (df[["id"] + columns]
                  .set_index(["id"], append = True).unstack(["id"])
                  .rolling(str(weekrange * 7) + "D", closed = "right")
                  .agg(["mean", "max", "min"])
                  .stack(1))
        df_tmp.columns = [str(x[0] + "_" + str(x[1]) + str(weekrange) + "week")
                          for x in df_tmp.columns.to_flat_index().values]
        df_tmp.index.set_names("id", 1, inplace = True)
        return df_tmp
    # return (df[["id"] + columns]
    #         .set_index(["id"], append = True).unstack(["id"])
    #         .rolling(str(weekrange * 7) + "D", closed = "right").mean().stack(["id"])
    #         .rename(columns = {x: x + "_mean" + str(weekrange) + "week" for x in columns}))
    demand = (df[["id"]].set_index(["id"], append = True)
              .join(demand_calc(weekrange = 1), how = "left")
              .join(demand_calc(weekrange = 2), how = "left")
              .join(demand_calc(weekrange = 4, columns = ["demand"]), how = "left")
              .join(demand_calc(weekrange = 12, columns = ["demand"]), how = "left")
              .join(demand_calc(weekrange = 48, columns = ["demand"]), how = "left")
              )
    demand = demand.drop(columns = ["snap_max1week", "snap_max2week", "snap_min1week", "snap_min2week"])

    # Rolling average with same weekday
    def demand_sameweekday_calc(weekrange, columns = ["demand"]):
        df_tmp = (df[["id"] + ["dayofweek"] + columns]
                .groupby(["dayofweek"])
                .apply(lambda x: (x[["id"] + columns].set_index(["id"], append = True).unstack(["id"])
                                  .rolling(str(weekrange * 7) + "D", closed = "right")
                                  .agg(["mean", "max", "min"])
                                  .stack(1)))
                .reset_index(["dayofweek"], drop = True))
        df_tmp.columns = [str(x[0] + "_" + str(x[1]) + str(weekrange) + "week_sameweekday")
                          for x in df_tmp.columns.to_flat_index().values]
        df_tmp.index.set_names("id", 1, inplace = True)
        return df_tmp
    demand_sameweekday = (df[["id"]].set_index(["id"], append = True)
                              .join(demand_sameweekday_calc(weekrange = 2), how = "left")
                              .join(demand_sameweekday_calc(weekrange = 4), how = "left")
                              .join(demand_sameweekday_calc(weekrange = 12), how = "left")
                              .join(demand_sameweekday_calc(weekrange = 48), how = "left")
                              )

    # Rolling average with same weekday by snap
    def demand_mean_sameweekday_samesnap_calc(weekrange, columns = ["demand"]):
        df_tmp = (df[["id"] + ["dayofweek", "snap"] + columns]
                  .groupby(["dayofweek", "snap"])
                  .apply(lambda x: (x[["id"] + columns].set_index(["id"], append = True).unstack(["id"])
                                    .rolling(weekrange, min_periods = 1).mean()
                                    .rename(columns = {x: x + "_mean" + str(weekrange) + "week_sameweekday_samesnap"
                                                       for x in columns})
                                    .stack(["id"])))
                  .reset_index(["dayofweek"], drop = True)
                  .unstack(["snap"]))
        df_tmp.columns = [str(x[0] + str(x[1])) for x in df_tmp.columns.to_flat_index().values]
        return df_tmp
    demand_mean_sameweekday_samesnap = (df[["id"]].set_index(["id"], append = True)
                                       .join(demand_mean_sameweekday_samesnap_calc(weekrange = 2), how = "left")
                                       .join(demand_mean_sameweekday_samesnap_calc(weekrange = 4), how = "left")
                                       .join(demand_mean_sameweekday_samesnap_calc(weekrange = 12), how = "left")
                                       .join(demand_mean_sameweekday_samesnap_calc(weekrange = 48), how = "left")
                                       )

    # Join ts features together, check and drop original demand
    df_tsfe = (df[["id"] + ["demand"]].set_index(["id"], append = True)
               .join(demand_lag, how = "left")
               .join(demand, how = "left")
               .reset_index())
    df_tsfe_sameweekday = (df[["id"] + ["demand"]].set_index(["id"], append = True)
                           .join(demand_sameweekday, how = "left")
                           .join(demand_mean_sameweekday_samesnap, how = "left")
                           .reset_index())
    #df_tmp = df_tsfe.reset_index().sort_values(by = ["id"] + ["date"])  # Check
    #df_tmp_sameweekday = df_tsfe_sameweekday.reset_index().sort_values(by = ["id"] + ["date"])  # Check
    del df, demand_lag, demand, demand_sameweekday, demand_mean_sameweekday_samesnap
    gc.collect()
    df_tsfe = df_tsfe.drop(columns = ["demand"])
    df_tsfe_sameweekday = df_tsfe_sameweekday.drop(columns = ["demand"])

    return df_tsfe, df_tsfe_sameweekday

# Run in parallel
tmp = datetime.now()
l_datasplit = [df.loc[df["id"].isin(x), ["date", "id", "demand", "snap", "dayofweek"]]
               for x in np.array_split(df["id"].unique(), 16)]
l_return = (Parallel(n_jobs = 8, max_nbytes = '1000M')(delayed(run_in_parallel)(x) for x in l_datasplit))
print(datetime.now() - tmp)
del l_datasplit
gc.collect()
df_tsfe = pd.concat([x[0] for x in l_return]).reset_index(drop = True)
df_tsfe_sameweekday = pd.concat([x[1] for x in l_return]).reset_index(drop = True)
del l_return
gc.collect()
df = reduce_mem_usage(df, float_convert = True)
df_tsfe = reduce_mem_usage(df_tsfe, float_convert = True)
df_tsfe_sameweekday = reduce_mem_usage(df_tsfe_sameweekday, float_convert = True)


########################################################################################################################
# Prepare final data
########################################################################################################################

# Remove Na-records from train
df = df.loc[((df["fold"] == "train") & (df["demand"].notna())) | (df["fold"] == "test")].reset_index(drop = True)
plt.close(fig="all")  # plt.close(plt.gcf())

# --- Save image ------------------------------------------------------------------------------------------------------

suffix = "" if n_sample is None else "_" + str(n_sample)
df.to_feather("df" + suffix + ".ftr")
df_tsfe.to_feather("df_tsfe" + suffix + ".ftr")
df_tsfe_sameweekday.to_feather("df_tsfe_sameweekday" + suffix + ".ftr")
df_help.to_feather("df_help" + suffix + ".ftr")

# Serialize
# with open("etl" + "_n" + str(n_sample) + "_ts.pkl" if n_sample is not None else "etl.pkl", "wb") as file:
#     pickle.dump({"df": df#,
#                  "df_tsfe": df_tsfe#,
#                  "df_tsfe_sameweekday": df_tsfe_sameweekday
#                 },
#                 file, protocol = 4)

print(datetime.now() - begin)

