
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from datetime import datetime
from datetime import timedelta
import gc
plt.ion(); matplotlib.use('TkAgg')
begin = datetime.now()

# Specific parameter
n_sample = None
n_jobs = 16
#horizon = 28
d_comb = {1: ["dummy"],
          2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
          6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
          10: ["item_id"], 11: ["item_id", "state_id"], 12: ["item_id", "store_id"]} # Aggregation levels

# Load results from etl
suffix = "" if n_sample is None else "_" + str(n_sample)
df = pd.read_feather("df" + suffix + ".ftr")
df_tsfe = pd.read_feather("df_tsfe" + suffix + ".ftr").set_index("date")
df_tsfe_sameweekday = pd.read_feather("df_tsfe_sameweekday" + suffix + ".ftr").set_index("date")
df_help = pd.read_feather("df_help" + suffix + ".ftr")

# with open("etl" + "_" + "n5000" + ".pkl", "rb") as file:
#     d_pick = pickle.load(file)
# df, df_tsfe, df_tsfe_sameweekday = d_pick["df"], d_pick["df_tsfe"], d_pick["df_tsfe_sameweekday"]


# --- Read metadata -------------------------------------------------------------------------------------------

df_meta = pd.read_excel("DATAMODEL_m5.xlsx")
df_meta_sub = df_meta.query("status == 'ready'") # Filter on "ready"


# ######################################################################################################################
# ETL
# ######################################################################################################################

# Train/Test fold: usually split by time
df.loc[df["year"] == 2011, "myfold"] = "util"
df.groupby("myfold")["date"].describe()
df["encode_flag"] = df["myfold"].map({"train": 0, "test": 0, "util": 1}).fillna(0)  # Used for encoding
df["fold_num"] = np.where(df["fold"] == "train", 0, 1)
df.fold_num.value_counts()


# --- Metric variables: Explore and adapt ------------------------------------------------------------------------------

# Define metric covariates
metr = df_meta_sub.query("h_dep == 'N' and modeltype == 'metr'")["variable"].values
print(df[metr].dtypes)

# Missings
print("\n\n misspct: \n", df[metr].isnull().mean().round(3).sort_values(ascending = False))

# Univariate Varimp
print("\n\n varimp_metr: \n", calc_imp(df.sample(n = int(1e5)), metr, target = "demand", target_type = "REGR"))

# Time/fold depedency
print("\n\n varimp_metr_fold: \n", calc_imp(df.sample(n = int(1e5)), metr, target = "fold_num"))


# --- Categorical variables: Explore and adapt -------------------------------------------------------------------------

# Define categorical covariates
cate = df_meta_sub.query("h_dep == 'N' and modeltype == 'cate'")["variable"].values
df = Convert(features = df[cate].dtypes.index.values[df[cate].dtypes != "object"], convert_to = "str").fit_transform(df)

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()
print(df[cate].nunique().sort_values(ascending = False))  # number of levels

# Univariate Varimp
print("\n\n varimp_cate: \n", calc_imp(df.sample(n = int(1e5)), cate, target = "demand", target_type = "REGR"))

# Time/fold depedency
print("\n\n varimp_cate_fold: \n", calc_imp(df.sample(n = int(1e5)), cate, target = "fold_num"))

# Create encoded features (for tree based models), i.e. numeric representation
enc = TargetEncoding(features = cate, encode_flag_column = "encode_flag", target = "demand",
                    remove_burned_data = False,
                    suffix = "").fit(df)
df = enc.transform(df)

# Adapt df_help with encoding
help_columns = ["cat_id","dept_id","item_id","state_id","store_id"]
df_help[help_columns] = df_help[help_columns].apply(lambda x: x.map(enc._d_map[x.name]))


# ######################################################################################################################
#  Add ts features
# ######################################################################################################################

df_test = pd.DataFrame()
for horizon in range(1,29):
    #horizon = 1
    print("HORIZON ", horizon)
    print(datetime.now() - begin)

    # ---  Join depending on horizon -----------------------------------------------------------------------------------
    tmp = datetime.now()
    df_h = (df.set_index(["date", "id"])
          .join(df_tsfe.shift(horizon, "D").set_index("id", append = True), how = "left")
          .join(df_tsfe_sameweekday.shift(((horizon - 1) // 7 + 1) * 7, "D").set_index("id", append = True),
                how = "left")
          .assign(demand_mean2week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                            x["demand_mean2week_sameweekday_samesnap0"],
                                                                            x["demand_mean2week_sameweekday_samesnap1"]))
          .assign(demand_mean4week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                            x["demand_mean4week_sameweekday_samesnap0"],
                                                                            x["demand_mean4week_sameweekday_samesnap1"]))
          .assign(demand_mean12week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                             x["demand_mean12week_sameweekday_samesnap0"],
                                                                             x["demand_mean12week_sameweekday_samesnap1"]))
          .assign(demand_mean48week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                             x["demand_mean48week_sameweekday_samesnap0"],
                                                                             x["demand_mean48week_sameweekday_samesnap1"]))
          .drop(columns = ['demand_mean2week_sameweekday_samesnap0', 'demand_mean2week_sameweekday_samesnap1',
                           'demand_mean4week_sameweekday_samesnap0', 'demand_mean4week_sameweekday_samesnap1',
                           'demand_mean12week_sameweekday_samesnap0', 'demand_mean12week_sameweekday_samesnap1',
                           'demand_mean48week_sameweekday_samesnap0', 'demand_mean48week_sameweekday_samesnap1'])
          .reset_index())
    print(datetime.now() - tmp)
    df_h = reduce_mem_usage(df_h, float_convert = True)

    # --- Prepare data and define final features -----------------------------------------------------------------------
    df_train = df_h.query("fold == 'train'").reset_index(drop = True)
    #df_test = df_h.query("fold == 'test'").reset_index(drop = True)
    df_test_h = (df_h.loc[df_h["date"] == df_h.query("fold == 'test'").date.min() + timedelta(days = horizon - 1)]
                 .reset_index(drop = True))
    del df_h
    gc.collect()
    metr = df_meta_sub.query("modeltype == 'metr'")["variable"].values
    cate = df_meta_sub.query("modeltype == 'cate'")["variable"].values
    all_features = np.concatenate([metr, cate])

    # --- Fit and Score ------------------------------------------------------------------------------------------------
    # Sample with weight
    df_train = (df_train
                .assign(weight_sales = lambda x: x["weight_sales"].pow(0.5))
                .sample(frac = 1, replace = True, weights = "weight_sales", random_state = 2)
            .reset_index(drop = True))
    # Fit
    lgb_param = dict(n_estimators = 3000, learning_rate = 0.04,  # TODO
                     num_leaves = 63, min_child_samples = 10,
                     colsample_bytree = 0.6, subsample = 1,
                     objective = "rmse",
                     n_jobs = n_jobs)
    fit = (lgbm.LGBMRegressor(**lgb_param)
           .fit(X = df_train[all_features],
                y = df_train["demand"],
                categorical_feature = cate.tolist()))


    # Score
    df_test_h["yhat"] = fit.predict(df_test_h[all_features])
    df_test = pd.concat([df_test, df_test_h])

df_test["yhat"] = np.where((df_test["sell_price_isna"] == 1) | (df_test["yhat"] < 0), 0.0001, df_test["yhat"] )



# --- Write submission -------------------------------------------------------------------------------------------------

df_tmp = df_test[["id", "d", "yhat"]].set_index(["id", "d"]).unstack("d").reset_index()
df_submit = pd.concat([pd.DataFrame(df_tmp.iloc[:, 1:29].values.round(5)).assign(id = df_tmp["id"]),
                       (pd.DataFrame(df_tmp.iloc[:, 1:29].values.round(5))
                        #.assign(id = df_tmp["id"].str.replace("validation", "evaluation")))])
                        .assign(id = df_tmp["id"].str.replace("evaluation", "validation")))])
df_submit.columns = ["F" + str(i) for i in range(1, 29)] + ["id"]
(pd.read_csv(dataloc + "sample_submission.csv")[["id"]]
 .merge(df_submit, on = "id", how = "left")
 .fillna(0)
 .to_csv("data/submit.csv", index = False))









