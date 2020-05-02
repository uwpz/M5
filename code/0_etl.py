
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
#  from scipy.stats.mstats import winsorize  # too slow
from datetime import datetime
#from statsmodels.graphics.tsaplots import plot_pacf
#from statsmodels.tsa.seasonal import seasonal_decompose

# Specific parameters
cutoff_corr = 0.1
cutoff_varimp = 0.52
ids = ["id"]
#n_stores = 3 #30  # number of stores to sample
#n_items = 10 #50  # number of items to sample


# ######################################################################################################################
#  ETL
# ######################################################################################################################

# --- Read data -----------------------------------------------------------------------------------------------------

# Items, stores, holidays
df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv").sample(n = 1000)
df_sales = pd.melt(df_sales_orig, id_vars = df_sales_orig.columns.values[:6],
                   var_name = "d", value_name = "demand")
df_calendar = pd.read_csv(dataloc + "calendar.csv", parse_dates=["date"])
df_prices = pd.read_csv(dataloc + "sell_prices.csv")
df = (df_sales
      .merge(df_calendar, how = "left", on = "d")
      .merge(df_prices, how = "left", on = ["store_id", "item_id", "wm_yr_wk"]))



'''
df[["id","demand"]].loc[df["demand"] > 0].groupby("id").median()["demand"].describe()
df["demand"].loc[df["sell_price"].notna()].describe()
df_prices.describe(include="all")
df_values = create_values_df(df_sales, 10)
df_sales_orig.describe(include="all")
plt.hist(np.log(df["demand"].loc[df["demand"]>0]), bins=50)
sum(df["demand"] == 0)/len(df)
df["demand"].loc[df["demand"]>0].value_counts().plot()
np.quantile(df["demand"].loc[df["demand"]>0], q = np.arange(0,1,0.01))
plt.hist(df["demand"].loc[df["demand"]>0], normed=True, cumulative=True, label='CDF',
         histtype='step', alpha=0.8, color='k', range = (0,10))
def cdf(x, plot=True, *args, **kwargs):
    x, y = sorted(x), np.arange(len(x)) / len(x)
    return plt.plot(x, y, *args, **kwargs) if plot else (x, y)
fig, ax = plt.subplots(1, 1)
ax = cdf(df["demand"].loc[df["demand"]>0])

tmp_cmap = mcolors.LinearSegmentedColormap.from_list("gr_bl_yl_rd",[(0.5, 0.5, 0.5, 0), "blue", "yellow", "red"])
sns.heatmap(df.iloc[:1000, 6:], cmap = tmp_cmap)

df_blub = df_prices.groupby(["store_id","item_id"]).std().query("sell_price > 0")
df_tmp = df_prices.loc[df_prices.item_id == "HOBBIES_1_108"]
sns.lineplot(x="wm_yr_wk", y="sell_price", hue="store_id", data=df_tmp)

df_tmp = df.loc[df["sales_isna"] == 0]
df_tmp["item_id"].value_counts()
fig, ax = plt.subplots(1,1)
sns.heatmap(df_tmp.loc[df_tmp["item_id"] == "FOODS_1_195"].groupby(["date", "id"])["anydemand"].mean().unstack("date"))
fig.tight_layout()
'''


# --- Transform -----------------------------------------------------------------------------------------------------

# Binary target
df["anydemand"] = np.where(df["demand"] > 0, 1, 0)

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

horizon = 1 #+ np.arange(15)
freq = 7
shift = freq * (1 + 2 * (horizon//(freq + 0.0001))) - horizon


# --- Shift the target (features will be leftjoined to it) -----------------------------------------------------------

df = df.set_index(["date"])
target = df[ids + ["demand", "anydemand"]]\
    .shift(-horizon, "D")\
    .rename(columns={"demand": "target", "anydemand": "anytarget"})\
    .set_index(ids, append=True)


# --- Basics --------------------------------------------------------------------------------------------------------

# Not time-based
df["sell_price_isna"] = np.where(df["sell_price"].isna(), 1, 0)

# Time-based
df["dayofweek"] = df["date"].index.dayofweek
df["weekend"] = np.where(df.dayofweek.isin([5, 6]), 1, 0)
df["dayofmonth"] = df["date"].dt.day
df["dayofyear"] = df["date"].dt.dayofyear
df["week"] = df["date"].dt.week
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
#df_train["id"] = np.arange(len(df_train)) + 1


# --- Time Series ----------------------------------------------------------------------------------------------------

# Lag values (might depend on horizon)
demand_lag1d = (df[ids + ["demand"]]
                .rename(columns = {"demand": "demand_lag1d"})
                .set_index(ids, append = True))
demand_lag7d = (df[ids + ["demand"]]
                .shift(shift, "D")
                .rename(columns = {"demand": "demand_lag7d"})
                .set_index(ids, append = True))
demand_avg7d = (df[ids + ["demand"]]
                .set_index(ids, append = True).unstack(ids)
                .rolling("6D").mean().stack(ids)
                .rename(columns = {"demand": "demand_avg7d"}))
demand_avg4sameweekdays = (df.groupby("dayofweek")
                           .apply(lambda x: x[ids + ["demand"]].set_index(ids, append = True).unstack(ids)
                                  .shift(shift, "D").rolling("21D").mean().stack(ids))
                           .reset_index("dayofweek", drop = True)
                           .rename(columns = {"demand": "demand_avg4sameweekdays"}))
demand_avg12sameweekdays = (df.groupby("dayofweek")
                            .apply(lambda x: x[ids + ["demand"]].set_index(ids, append=True).unstack(ids)
                                   .shift(shift, "D").rolling("77D").mean().stack(ids))
                            .reset_index("dayofweek", drop = True)
                            .rename(columns = {"demand": "demand_avg12sameweekdays"}))

# Join ts features together
df_tsfe = (demand_lag1d
           .join(demand_lag7d, how="left")
           .join(demand_avg7d, how="left")
           .join(demand_avg4sameweekdays, how="left")
           .join(demand_avg12sameweekdays, how="left"))


# --- Join all  --------------------------------------------------------------------------------------------------

df_train = (target
            .join(df.set_index(ids, append = True).drop(columns = ["demand", "anydemand"]), how = "left")
            .join(df_tsfe, how = "left")
            .reset_index())
#df["id"] = np.arange(len(df)) + 1


# --- Read metadata (Project specific) -----------------------------------------------------------------------------

df_meta = pd.read_excel(dataloc + "DATAMODEL_favorita.xlsx")

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.loc[df_meta["status"] == "ready", "variable"].values, df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready"])]


# ######################################################################################################################
#  Plot
# ######################################################################################################################

'''
ts = df.groupby("date")["target"].mean()

# STL decomposition
sdec1 = seasonal_decompose(ts, freq=7)
sdec1.plot()
sdec2 = seasonal_decompose(ts - sdec1.seasonal, freq=30)
sdec2.plot()

# Pacf plot
plot_pacf(ts, lags=40)
plot_pacf(ts - sdec1.seasonal, lags=40)
plot_pacf(ts - sdec1.seasonal - sdec2.seasonal, lags = 40)

# ts.rolling("90D", min_periods=90).mean().plot()
'''


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Save image ------------------------------------------------------------------------------------------------------
plt.close(fig="all")  # plt.close(plt.gcf())

# Serialize
with open("0_etl_h" + str(horizon) + ".pkl", "wb") as file:
    pickle.dump({"df": df,
                 "df_meta_sub": df_meta_sub},
                file)


