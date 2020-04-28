
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
#ids = ["store_nbr", "item_nbr"]
#n_stores = 3 #30  # number of stores to sample
#n_items = 10 #50  # number of items to sample


# ######################################################################################################################
#  ETL
# ######################################################################################################################

# --- Read data -----------------------------------------------------------------------------------------------------

# Items, stores, holidays
df_sales_orig = pd.read_csv(dataloc + "sales_train_validation.csv").iloc[:, :200]
#tmp_cmap = mcolors.LinearSegmentedColormap.from_list("gr_bl_yl_rd",[(0.5, 0.5, 0.5, 0), "blue", "yellow", "red"])
#sns.heatmap(df_sales_orig.iloc[:1000, 6:], cmap = tmp_cmap)
df_sales = pd.melt(df_sales_orig, id_vars = df_sales_orig.columns.values[:6],
                   var_name = "d", value_name = "demand")
df_calendar = pd.read_csv(dataloc + "calendar.csv", parse_dates=["date"])
df_prices = pd.read_csv(dataloc + "sell_prices.csv")
df = (df_sales
      .merge(df_calendar, how = "left", on = "d")
      .merge(df_prices, how = "left", on = ["store_id", "item_id", "wm_yr_wk"]))
'''
df_prices.describe(include="all")
df_values = create_values_df(df_sales, 10)
df_sales_orig.describe(include="all")
plt.boxplot((df["demand"].loc[df["demand"]>0]))
sum(df["demand"] == 0)/len(df)
df["demand"].value_counts().plot()

'''

# Train data
'''
tmp = datetime.now()
df_train = pd.read_csv(dataloc + "train.csv", 
    dtype = {'onpromotion': bool}, parse_dates = ["date"],
    skiprows = range(1, 66458909)  # starting 2016-01-01
).set_index("date")
print(str(datetime.now()- tmp))
df_train.to_pickle("df_train.pkl")
'''
tmp = datetime.now()
df_train = pd.read_pickle("df_train.pkl")
print(str(datetime.now() - tmp))
np.random.seed(123)
stores = df_stores["store_nbr"].sample(n_stores).values
items = df_items["item_nbr"].sample(n_items).values
df_train = df_train[df_train["store_nbr"].isin(stores) & df_train["item_nbr"].isin(items)]

# Create store-holiday table
df_tmp = df_holiday.loc[df_holiday["transferred"] == False, ["date", "locale", "locale_name"]]
df_storesholiday = pd.concat(
    [pd.merge(df_tmp.loc[df_tmp["locale"] == "National", ["date", "locale_name"]],
              df_stores[["store_nbr", "country"]],
              left_on="locale_name", right_on="country",
              how="left")[["store_nbr", "date"]],
     pd.merge(df_tmp.loc[df_tmp["locale"] == "Regional", ["date", "locale_name"]],
              df_stores[["store_nbr", "state"]],
              left_on="locale_name", right_on="state",
              how="left")[["store_nbr", "date"]],
     pd.merge(df_tmp.loc[df_tmp["locale"] == "Local", ["date", "locale_name"]],
              df_stores[["store_nbr", "city"]],
              left_on="locale_name", right_on="city",
              how="left")[["store_nbr", "date"]]])\
    .drop_duplicates()\
    .assign(holiday=1)


# --- Transform -----------------------------------------------------------------------------------------------------
# Make onpromotion numeric
df_train.onpromotion = df_train.onpromotion.astype("int")

# Set negative unit_sales to 0
df_train.loc[df_train["unit_sales"] <= 0, "unit_sales"] = 0

# Some checks
df_train.groupby(ids)["unit_sales"].count().plot(kind="hist", bins=50)
df_tmp = df_train.reset_index().groupby(ids)["date"].min()

# Fill up gaps with 0
df_train = df_train.groupby(ids)["unit_sales", "onpromotion"].resample("D").asfreq(fill_value=0).reset_index(ids)

# Transform to log and scale unit_sales to sales
df_train["unit_sales_log"] = np.log(df_train.unit_sales.values + 1)
df_train["mean_unit_sales_log"] = df_train.groupby("item_nbr")["unit_sales_log"].transform("mean")
df_train["std_unit_sales_log"] = df_train.groupby("item_nbr")["unit_sales_log"].transform("std")
df_train["std_unit_sales_log"].replace(0, np.median(df_train["std_unit_sales_log"]), inplace=True)
df_train["zsales_log"] = (df_train["unit_sales_log"] - df_train["mean_unit_sales_log"]) / df_train["std_unit_sales_log"]
df_train["zsales_log"].describe()
df_train["zsales_log"].plot(kind="hist", bins=50)

# Winsorize
tmp_perc = np.nanpercentile(df_train["zsales_log"], np.array([0.5, 99.5]))
df_train.loc[df_train["zsales_log"] < tmp_perc[0], "zsales_log"] = tmp_perc[0]
df_train.loc[df_train["zsales_log"] > tmp_perc[1], "zsales_log"] = tmp_perc[1]

# Recalc unit_sales_log and scale again with winsorized data
df_train["unit_sales_log"] = df_train.zsales_log * df_train.std_unit_sales_log + df_train.mean_unit_sales_log
df_train["mean_unit_sales_log"] = df_train.groupby("item_nbr")["unit_sales_log"].transform("mean")
df_train["std_unit_sales_log"] = df_train.groupby("item_nbr")["unit_sales_log"].transform("std")
df_train["std_unit_sales_log"].replace(0, np.median(df_train["std_unit_sales_log"]), inplace=True)
df_train["zsales_log"] = (df_train["unit_sales_log"] - df_train["mean_unit_sales_log"]) / df_train["std_unit_sales_log"]
df_train["zsales_log"].plot(kind="hist", bins=50)
df_items = df_items.merge(df_train[["item_nbr", "mean_unit_sales_log", "std_unit_sales_log"]]\
                            .reset_index(drop=True)\
                            .drop_duplicates(),
                          on="item_nbr", how="left")
df_train.drop(columns=["unit_sales_log", "mean_unit_sales_log", "std_unit_sales_log"],
              inplace=True)

# Set fold
df_train["fold"] = "train"
df_train["myfold"] = "train"
df_train.loc["2017-07-31":, "myfold"] = "test"
df_train.myfold.value_counts()
df_train.reset_index().groupby(ids)["date"].max()

# Impute onpromotion
df_train.onpromotion.describe()
df_train.loc[df_train.onpromotion.isnull(), "onpromotion"] = 0


# ######################################################################################################################
#  Feature engineering
# ######################################################################################################################

horizon = 1 #+ np.arange(15)
freq = 7
shift = freq * (1 + 2 * (horizon//(freq + 0.0001))) - horizon


# --- Shift the target (features will be leftjoined to it) ------------------------------------------------------------
target = df_train[ids + ["zsales_log"]]\
    .shift(-horizon, "D")\
    .rename(columns={"zsales_log": "target"})\
    .set_index(ids, append=True)


# --- Basics --------------------------------------------------------------------------------------------------------
df_train["dayofweek"] = df_train.index.dayofweek
df_train["weekend"] = np.where(df_train.dayofweek.isin([5, 6]), 1, 0)
df_train["dayofmonth"] = df_train.index.day
df_train["dayofyear"] = df_train.index.dayofyear
df_train["week"] = df_train.index.week
df_train["month"] = df_train.index.month
df_train["year"] = df_train.index.year
df_train["id"] = np.arange(len(df_train)) + 1


# --- dayofpromotion -----------------------------------------------------------------------------------
df_onprom = df_train[ids + ["onpromotion"]].set_index(ids, append=True)
df_onprom["lag_onpromotion"] = df_onprom.reset_index(ids).shift(1, "D").set_index(ids, append=True)
df_onprom["flag_promotion"] = (df_onprom["onpromotion"] != df_onprom["lag_onpromotion"]).astype("int")
df_onprom["group_promotion"] = df_onprom["flag_promotion"].cumsum()
df_onprom["day_ofpromotion"] = df_onprom.groupby(ids + ["group_promotion"])["onpromotion"].cumsum()
df_onprom = df_onprom["day_ofpromotion"]


# --- Time Series ------------------------------------------------------------------------------------------------------
# Lag values (might depend on horizon)
sales_lag1d = df_train[ids + ["zsales_log", "onpromotion"]]\
    .rename(columns={"zsales_log": "zsales_log_lag1d", "onpromotion": "onpromotion_lag1d"})\
    .set_index(ids, append=True)
sales_lag7d = df_train[ids + ["zsales_log", "onpromotion"]]\
    .shift(shift, "D")\
    .rename(columns={"zsales_log": "zsales_log_lag7d", "onpromotion": "onpromotion_lag7d"})\
    .set_index(ids, append=True)
sales_avg7d = df_train[ids + ["zsales_log", "onpromotion"]]\
    .set_index(ids, append=True).unstack(ids)\
    .rolling("6D").mean().stack(ids)\
    .rename(columns={"zsales_log": "zsales_log_avg7d", "onpromotion": "onpromotion_avg7d"})
sales_avg4sameweekdays = df_train.groupby("dayofweek").apply(
    lambda x: x[ids + ["zsales_log", "onpromotion"]].set_index(ids, append=True).unstack(ids)
              .shift(shift, "D").rolling("21D").mean().stack(ids)) \
    .reset_index("dayofweek", drop=True) \
    .rename(columns={"zsales_log": "zsales_log_avg4sameweekdays", "onpromotion": "onpromotion_avg4sameweekdays"})
sales_avg12sameweekdays = df_train.groupby("dayofweek").apply(
    lambda x: x[ids + ["zsales_log", "onpromotion"]].set_index(ids, append=True).unstack(ids)
              .shift(shift, "D").rolling("77D").mean().stack(ids)) \
    .reset_index("dayofweek", drop=True) \
    .rename(columns={"zsales_log": "zsales_log_avg12sameweekdays", "onpromotion": "onpromotion_avg12sameweekdays"})

# Join ts features together
df_tsfe = sales_lag1d
df_tsfe = df_tsfe.join(sales_lag7d, how="left")
df_tsfe = df_tsfe.join(sales_avg7d, how="left")
df_tsfe = df_tsfe.join(sales_avg4sameweekdays, how="left")
df_tsfe = df_tsfe.join(sales_avg12sameweekdays, how="left")


# --- Join all  --------------------------------------------------------------------------------------------------
df = target\
    .join(df_train.set_index(ids, append=True)
                  .drop(columns=["zsales_log"]),
          how="left")\
    .join(df_onprom, how="left")\
    .join(df_tsfe, how="left")\
    .join(df_items.set_index(["item_nbr"]), how="left")\
    .join(df_stores.set_index(["store_nbr"]), how="left")\
    .join(df_storesholiday.set_index(["date", "store_nbr"]), how="left").fillna({"holiday": 0})\
    .reset_index()
df["id"] = np.arange(len(df)) + 1


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


