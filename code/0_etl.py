
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
#  from scipy.stats.mstats import winsorize  # too slow
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.seasonal import seasonal_decompose

# Specific parameters
cutoff_corr = 0.1
cutoff_varimp = 0.52
ids = ["id"]
#n_stores = 3 #30  # number of stores to sample
#n_items = 10 #50  # number of items to sample

#plt.ioff(); matplotlib.use('Agg')
# plt.ion(); matplotlib.use('TkAgg')


# ######################################################################################################################
#  ETL
# ######################################################################################################################

# --- Read data -----------------------------------------------------------------------------------------------------

# Items, stores, holidays
df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv").sample(n = 300, random_state = 1)
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
df_tmp = df_calendar.loc[df_calendar.holiday_name.notna()]
df_prices = pd.read_csv(dataloc + "sell_prices.csv")
df = (df_sales
      .merge(df_calendar.drop(columns = ["weekday", "wday", "month", "year",
                                         "event_name_1", "event_type_1", "event_name_2", "event_type_2"]),
             how = "left", on = "d")
      .merge(df_prices, how = "left", on = ["store_id", "item_id", "wm_yr_wk"]))

# Important columns
df["anydemand"] = np.where(df["demand"] > 0, 1, 0)
df["sell_price_isna"] = np.where(df["sell_price"].isna(), 1, 0)
df["snap"] = np.where(df["state_id"] == "CA", df["snap_CA"],
                      np.where(df["state_id"] == "TX", df["snap_TX"], df["snap_WI"]))
df.drop(columns = ["snap_CA", "snap_TX", "snap_WI"], inplace = True)

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

# Winsorize and add median and iqr per id
df = (df.merge((df[["id","demand"]].loc[df["demand"] > 0].groupby("id")["demand"]
                .apply(lambda x: x.quantile(q = [0.25, 0.75, 0.5, 0.95]))
                .unstack()
                .assign(demand_iqr = lambda x: x[0.75] - x[0.25]).drop(columns = [0.75, 0.25])
                .reset_index()
                .rename(columns = {0.5: "demand_median", 0.95: "demand_upperlimit"})),
               how = "left")
      .assign(demand = lambda x: x[["demand", "demand_upperlimit"]].min(axis = 1))
      .drop(columns = ["demand_upperlimit"]))

# Adapt demand due to missing sell_price and xmas outlier
df.loc[df["sell_price_isna"] == 1, ["demand", "anydemand"]] = np.nan
df.loc[(df["date"].dt.day == 25) & (df["date"].dt.month == 12), ["demand", "anydemand"]] = np.nan

# Set fold
df["fold"] = "train"
df["myfold"] = np.where(df["date"] >= "2016-03-28", "test", "train")
df.groupby("myfold")["date"].nunique()


# Impute onpromotion
#df_train.onpromotion.describe()
#df_train.loc[df_train.onpromotion.isnull(), "onpromotion"] = 0


# ######################################################################################################################
#  Feature engineering
# ######################################################################################################################

# --- Basics --------------------------------------------------------------------------------------------------------

# Not time-based
# tbd

# Time-based
df["dayofweek"] = df["date"].dt.dayofweek
df["weekend"] = np.where(df.dayofweek.isin([5, 6]), 1, 0)
df["dayofmonth"] = df["date"].dt.day
df["dayofyear"] = df["date"].dt.dayofyear
df["week"] = df["date"].dt.week
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year


# --- Time Series ----------------------------------------------------------------------------------------------------

# Set index
df = df.set_index(["date"])

# Parameter settings
horizon = 8

# Lag values (might depend on horizon)
def demand_lag_horizon_calc(shift = 0):
    return (df[ids + ["demand"]]
            .shift(horizon + shift, "D")
            .rename(columns = {"demand": "demand_lag_horizon_plus" + str(shift)})
            .set_index(ids, append = True))
demand_lag_horizon = (df[ids + ["demand", "snap"]]
                      .shift(horizon, "D")
                      .rename(columns = {"demand": "demand_lag_horizon", "snap": "snap_lag_horizon"})
                      .set_index(ids, append = True)
                      .join(demand_lag_horizon_calc(shift = 1), how = "left")
                      .join(demand_lag_horizon_calc(shift = 2), how = "left")
                      .join(demand_lag_horizon_calc(shift = 3), how = "left")
                      .join(demand_lag_horizon_calc(shift = 4), how = "left")
                      .join(demand_lag_horizon_calc(shift = 5), how = "left")
                      .join(demand_lag_horizon_calc(shift = 6), how = "left")
                      )

# Lag same weekday
demand_lag_sameweekday = (df[ids + ["demand","snap"]]
                          .shift(((horizon - 1) // 7 + 1) * 7, "D")
                          .rename(columns = {"demand": "demand_lag_sameweekday", "snap": "snap_lag_sameweekday"})
                          .set_index(ids, append = True))

# Rolling average
def demand_avg_calc(weekrange = 1, suffix = "1week"):
    return (df[ids + ["demand"]]
            .shift(horizon, "D")
            .set_index(ids, append = True).unstack(ids)
            .rolling(str(weekrange * 7) + "D", closed = "right").mean().stack(ids)
            .rename(columns = {"demand": "demand_avg" + str(weekrange) + "week"}))
demand_avg = (df[ids + ["demand", "snap"]]
              .shift(horizon, "D")
              .set_index(ids, append = True).unstack(ids)
              .rolling("7D", closed = "right").mean().stack(ids)
              .rename(columns = {"demand": "demand_avg1week", "snap": "snap_avg1week"})
              .join(demand_avg_calc(weekrange = 2), how = "left")
              .join(demand_avg_calc(weekrange = 4), how = "left")
              .join(demand_avg_calc(weekrange = 12), how = "left")
              .join(demand_avg_calc(weekrange = 48), how = "left")
              )

# Rolling average with same weekday
def demand_avg_sameweekday_calc(weekrange = 4):
    return (df[ids + ["demand"] + ["dayofweek"]]
            .shift(((horizon - 1) // 7 + 1) * 7, "D")
            .groupby("dayofweek")
            .apply(lambda x: x[ids + ["demand"]].set_index(ids, append = True).unstack(ids)
                   .rolling(str(weekrange * 7) + "D", closed = "right").mean().stack(ids))
            .reset_index("dayofweek", drop = True)
            .rename(columns = {"demand": "demand_avg" + str(weekrange) + "week_sameweekday"}))
demand_avg_sameweekday = (df[ids].set_index(ids, append = True)
                           .join(demand_avg_sameweekday_calc(weekrange = 2), how = "left")
                           .join(demand_avg_sameweekday_calc(weekrange = 4), how = "left")
                           .join(demand_avg_sameweekday_calc(weekrange = 12), how = "left")
                           .join(demand_avg_sameweekday_calc(weekrange = 48), how = "left")
                           )

# Rolling average with same weekday and same snap
def demand_avg_sameweekday_samesnap_calc(weekrange = 4):
    return (df[ids + ["demand"] + ["dayofweek", "snap"]]
            .shift(((horizon - 1) // 7 + 1) * 7, "D")
            .groupby(["dayofweek", "snap"])
            .apply(lambda x: x[ids + ["demand"]].set_index(ids, append = True).unstack(ids)
                   .rolling(str(weekrange * 7) + "D",  closed = "right").mean().stack(ids))
            .reset_index(["dayofweek"], drop = True)
            .reset_index("snap").set_index("snap", append = True)
            .rename(columns = {"demand": "demand_avg" + str(weekrange) + "week_sameweekday_samesnap"}))
demand_avg_sameweekday_samesnap = (df[ids + ["snap"]].set_index(ids + ["snap"], append = True)
                                   .join(demand_avg_sameweekday_samesnap_calc(weekrange = 2), how = "left")
                                   .join(demand_avg_sameweekday_samesnap_calc(weekrange = 4), how = "left")
                                   .join(demand_avg_sameweekday_samesnap_calc(weekrange = 12), how = "left")
                                   .join(demand_avg_sameweekday_samesnap_calc(weekrange = 48), how = "left")
                                   ).reset_index("snap", drop = True)

# Join ts features together, check and drop original demand
df_tsfe = (df[ids + ["demand"]].set_index(ids, append = True)
           .join(demand_lag_horizon, how = "left")
           .join(demand_lag_sameweekday, how = "left")
           .join(demand_avg, how = "left")
           .join(demand_avg_sameweekday, how = "left")
           .join(demand_avg_sameweekday_samesnap, how = "left"))
df_tmp = df_tsfe.reset_index().sort_values(by = ids + ["date"])  # Check
df_tsfe = df_tsfe.drop(columns = ["demand"])


# --- Join all and remove missing target------------------------------------------------------------------------

df_train = df.set_index(ids, append = True).join(df_tsfe, how = "left").reset_index()
df_train = df_train.loc[df_train["demand"].notna()]


# --- Check metadata -------------------------------------------------------------------------------------------

df_meta = pd.read_excel("DATAMODEL_m5.xlsx")

# Check
print(setdiff(df_train.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.loc[df_meta["status"] == "ready", "variable"].values, df_train.columns.values))
print(setdiff(df_train.columns.values, df_meta.loc[df_meta["status"] == "ready", "variable"].values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready"])]


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Save image ------------------------------------------------------------------------------------------------------
plt.close(fig="all")  # plt.close(plt.gcf())

# Serialize
with open("0_etl_h" + str(horizon) + ".pkl", "wb") as file:
    pickle.dump({"df_train": df_train,
                 "df_meta_sub": df_meta_sub},
                file)


