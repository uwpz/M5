
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
ids = ["id"]
n_sample = 10000
n_jobs = 14
plt.ioff(); matplotlib.use('Agg')
# plt.ion(); matplotlib.use('TkAgg')


# ######################################################################################################################
#  ETL
# ######################################################################################################################

# --- Read data and 1st FE ---------------------------------------------------------------------------------------------

# Sales, calendar, prices
if n_sample is None:
    df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv")
else:
    df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv").sample(n = int(n_sample), random_state = 1)
df_sales_orig[["d_" + str(x) for x in range(1914, 1942)]] = pd.DataFrame([[np.nan for x in range(1914, 1942)]])
df_sales = pd.melt(df_sales_orig, id_vars = df_sales_orig.columns.values[:6],
                   var_name = "d", value_name = "demand")
holidays = ["ValentinesDay","StPatricksDay","Easter","Mother's day","Father's day","IndependenceDay",
            "Halloween","Thanksgiving","Christmas", "NewYear"]
df_calendar = (pd.read_csv(dataloc + "calendar.csv", parse_dates=["date"])
               .assign(event_name = lambda x: np.where(x["event_name_2"].isin(["Easter", "Cinco De Mayo"]),
                                                       x["event_name_2"], x["event_name_1"]))
               .assign(event_type = lambda x: np.where(x["event_name_2"].isin(["Easter", "Cinco De Mayo"]),
                                                       x["event_type_2"], x["event_type_1"]))
               .assign(event = lambda x: x["event_name"].notna().astype("int"))
               .assign(next_event = lambda x: x["event_name"].fillna(method = "bfill"))
               .assign(prev_event = lambda x: x["event_name"].fillna(method = "ffill"))
               .assign(days_before_event = lambda x: x.groupby("next_event").cumcount() + 1)
               .assign(days_after_event = lambda x: x.groupby("prev_event").cumcount() + 1)
               .assign(holiday_name = lambda x: np.where(x["event_name_1"].isin(holidays), x["event_name"], np.nan))
               .assign(holiday_name = lambda x: np.where(x["event_name_2"].isin(holidays),
                                                         x["event_name_2"], x["holiday_name"]))
               .assign(holiday = lambda x: x["holiday_name"].notna().astype("int"))
               .assign(next_holiday = lambda x: x["holiday_name"].fillna(method = "bfill"))
               .assign(prev_holiday = lambda x: x["holiday_name"].fillna(method = "ffill"))
               .assign(days_before_holiday = lambda x: x.groupby("next_holiday").cumcount() + 1)
               .assign(days_after_holiday = lambda x: x.groupby("prev_holiday").cumcount() + 1)
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

# Important columns
df["anydemand"] = np.where(df["demand"] > 0, 1, np.where(df["demand"].notna(), 0, np.nan))
df["sell_price_isna"] = np.where(df["sell_price"].isna(), 1, 0)  # no sales if ==1
df["snap"] = np.where(df["state_id"] == "CA", df["snap_CA"],
                      np.where(df["state_id"] == "TX", df["snap_TX"], df["snap_WI"]))  # compress snap
df = df.drop(columns = ["snap_CA", "snap_TX", "snap_WI"])

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

# --- Transform -----------------------------------------------------------------------------------------------------

del df_sales_orig, df_sales, df_calendar, df_prices
gc.collect()

# Winsorize (and add median and iqr per id: not used so far)
df = (df.merge((df[["id","demand"]].loc[df["demand"] > 0].groupby("id")["demand"]
                .apply(lambda x: x.quantile(q = [0.25, 0.75, 0.5, 0.95]))
                .unstack()
                .assign(demand_iqr = lambda x: x[0.75] - x[0.25]).drop(columns = [0.75, 0.25])
                .reset_index()
                .rename(columns = {0.5: "demand_median", 0.95: "demand_upperlimit"})),
               how = "left")
      .assign(demand = lambda x: np.where(x["demand"].isna(), np.nan, x[["demand", "demand_upperlimit"]].min(axis = 1)))
      .drop(columns = ["demand_upperlimit"]))

# Adapt demand due to missing sell_price and xmas outlier
df.loc[df["sell_price_isna"] == 1, ["demand", "anydemand"]] = np.nan
df.loc[df["holiday_name"] == "Christmas", ["demand", "anydemand"]] = np.nan

# Set fold
df["fold"] = np.where(df["date"] >= "2016-04-25", "test", "train")
#df.groupby("fold")["date"].nunique()
df["myfold"] = np.where(df["date"] >= "2016-04-25", None, np.where(df["date"] >= "2016-03-28", "test", "train"))
#df.groupby("myfold")["date"].nunique()
df.myfold.describe()


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


# --- Advanced ----------------------------------------------------------------------------------------------------

def run_in_parallel(df):

    # Set index
    df = df.set_index(["date"])

    # Lag values (might depend on horizon)
    def demand_lag_calc(shift, ids = ["id"], columns = ["demand", "snap"]):
        return (df[ids + columns]
                .shift(shift, "D")
                .rename(columns = {x: x + "_lag" + str(shift) for x in columns})
                .set_index(ids, append = True))
    demand_lag = (df[ids].set_index(ids, append = True)
                  .join(demand_lag_calc(shift = 0), how = "left")
                  .join(demand_lag_calc(shift = 1), how = "left")
                  .join(demand_lag_calc(shift = 2), how = "left")
                  .join(demand_lag_calc(shift = 3), how = "left")
                  .join(demand_lag_calc(shift = 4), how = "left")
                  .join(demand_lag_calc(shift = 5), how = "left")
                  .join(demand_lag_calc(shift = 6), how = "left")
                  )
    #df_tmp = demand_lag.reset_index().sort_values(by = ids + ["date"])  # Check

    # Lag same weekday
    # demand_lag_sameweekday = (df[ids + ["demand","snap"]]
    #                           .shift(((horizon - 1) // 7 + 1) * 7, "D")
    #                           .rename(columns = {"demand": "demand_lag_sameweekday", "snap": "snap_lag_sameweekday"})
    #                           .set_index(ids, append = True))

    # Rolling average
    def demand_avg_calc(weekrange, ids = ["id"], columns = ["demand", "snap"]):
        return (df[ids + columns]
                .set_index(ids, append = True).unstack(ids)
                .rolling(str(weekrange * 7) + "D", closed = "right").mean().stack(ids)
                .rename(columns = {x: x + "_avg" + str(weekrange) + "week" for x in columns}))
    demand_avg = (df[ids].set_index(ids, append = True)
                  .join(demand_avg_calc(weekrange = 1), how = "left")
                  .join(demand_avg_calc(weekrange = 2), how = "left")
                  .join(demand_avg_calc(weekrange = 4, columns = ["demand"]), how = "left")
                  .join(demand_avg_calc(weekrange = 12, columns = ["demand"]), how = "left")
                  .join(demand_avg_calc(weekrange = 48, columns = ["demand"]), how = "left")
                  )

    # Rolling average with same weekday
    def demand_avg_sameweekday_calc(weekrange, ids = ["id"], columns = ["demand"]):
        return (df[ids + ["dayofweek"] + columns]
                .groupby(["dayofweek"])
                .apply(lambda x: (x[ids + columns].set_index(ids, append = True).unstack(ids)
                                  .rolling(str(weekrange * 7) + "D", closed = "right").mean()
                                  #.rolling(weekrange, min_periods = 1).mean()
                                  .rename(columns = {x: x + "_avg" + str(weekrange) + "week_sameweekday"
                                                     for x in columns})
                                  .stack(ids)))
                .reset_index(["dayofweek"], drop = True))
    demand_avg_sameweekday = (df[ids].set_index(ids, append = True)
                              .join(demand_avg_sameweekday_calc(weekrange = 2), how = "left")
                              .join(demand_avg_sameweekday_calc(weekrange = 4), how = "left")
                              .join(demand_avg_sameweekday_calc(weekrange = 12), how = "left")
                              .join(demand_avg_sameweekday_calc(weekrange = 48), how = "left")
                              )

    # Rolling average with same weekday by snap
    def demand_avg_sameweekday_snap_calc(weekrange, ids = ["id"], columns = ["demand"]):
        df_tmp = (df[ids + ["dayofweek", "snap"] + columns]
                .groupby(["dayofweek", "snap"])
                .apply(lambda x: (x[ids + columns].set_index(ids, append = True).unstack(ids)
                                  .rolling(weekrange, min_periods = 1).mean()
                                  .rename(columns = {x: x + "_avg" + str(weekrange) + "week_sameweekday_samesnap"
                                                     for x in columns})
                                  .stack(ids)))
                .reset_index(["dayofweek"], drop = True)
                .unstack(["snap"]))
        df_tmp.columns = [str(x[0] + str(x[1])) for x in df_tmp.columns.to_flat_index().values]
        return df_tmp
    demand_avg_sameweekday_snap = (df[ids].set_index(ids, append = True)
                                       .join(demand_avg_sameweekday_snap_calc(weekrange = 2), how = "left")
                                       .join(demand_avg_sameweekday_snap_calc(weekrange = 4), how = "left")
                                       .join(demand_avg_sameweekday_snap_calc(weekrange = 12), how = "left")
                                       .join(demand_avg_sameweekday_snap_calc(weekrange = 48), how = "left")
                                       )

    # Join ts features together, check and drop original demand
    df_tsfe = (df[ids + ["demand"]].set_index(ids, append = True)
               .join(demand_lag, how = "left")
               .join(demand_avg, how = "left")
               .reset_index())
    df_tsfe_sameweekday = (df[ids + ["demand"]].set_index(ids, append = True)
                           .join(demand_avg_sameweekday, how = "left")
                           .join(demand_avg_sameweekday_snap, how = "left")
                           .reset_index())
    #df_tmp = df_tsfe.reset_index().sort_values(by = ids + ["date"])  # Check
    #df_tmp_sameweekday = df_tsfe_sameweekday.reset_index().sort_values(by = ids + ["date"])  # Check
    del df, demand_lag, demand_avg, demand_avg_sameweekday, demand_avg_sameweekday_snap
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

# Serialize
# with open("etl" + "_n" + str(n_sample) + "_ts.pkl" if n_sample is not None else "etl.pkl", "wb") as file:
#     pickle.dump({"df": df#,
#                  "df_tsfe": df_tsfe#,
#                  "df_tsfe_sameweekday": df_tsfe_sameweekday
#                 },
#                 file, protocol = 4)


