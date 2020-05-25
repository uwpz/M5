
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from datetime import datetime
import gc
plt.ion(); matplotlib.use('TkAgg')

# Specific parameter
n_sample = 1000
horizon = 28
n_jobs = 4

# Load results from etl
suffix = "" if n_sample is None else "_" + str(n_sample)
df = pd.read_feather("df" + suffix + ".ftr")
df_tsfe = pd.read_feather("df_tsfe" + suffix + ".ftr").set_index("date")
df_tsfe_sameweekday = pd.read_feather("df_tsfe_sameweekday" + suffix + ".ftr").set_index("date")
# with open("etl" + "_" + "n5000" + ".pkl", "rb") as file:
#     d_pick = pickle.load(file)
# df, df_tsfe, df_tsfe_sameweekday = d_pick["df"], d_pick["df_tsfe"], d_pick["df_tsfe_sameweekday"]


# --- Check metadata -------------------------------------------------------------------------------------------

df_meta = pd.read_excel("DATAMODEL_m5.xlsx")

# Check
print(setdiff(df.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.loc[(df_meta["status"] == "ready") & (df_meta["h_dep"] == 'N'), "variable"].values,
              df.columns.values))

# Filter on "ready"
df_meta_sub = df_meta.query("status == 'ready'")


# ######################################################################################################################
# ETL
# ######################################################################################################################

# Train/Test fold: usually split by time
df.loc[df["year"] == 2011, "myfold"] = "util"
df.groupby("myfold")["date"].describe()
df["encode_flag"] = df["myfold"].map({"train": 0, "test": 0, "util": 1})  # Used for encoding
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

# Create encoded features (for tree based models), i.e. numeric representation
df = TargetEncoding(features = cate, encode_flag_column = "encode_flag", target = "demand",
                    remove_burned_data = True,
                    suffix = "").fit_transform(df)

# Univariate Varimp
print("\n\n varimp_cate: \n", calc_imp(df.sample(n = int(1e5)), cate, target = "demand", target_type = "REGR"))

# Time/fold depedency
print("\n\n varimp_cate_fold: \n", calc_imp(df.sample(n = int(1e5)), cate, target = "fold_num"))


# ######################################################################################################################
#  Add ts features
# ######################################################################################################################

# ---  Join depending on horizon ---------------------------------------------------------------------------------------

tmp = datetime.now()
df = (df.set_index(["date", "id"])
      .join(df_tsfe.shift(horizon, "D").set_index("id", append = True), how = "left")
      .join(df_tsfe_sameweekday.shift(((horizon - 1) // 7 + 1) * 7, "D").set_index("id", append = True),
            how = "left")
      .assign(demand_avg2week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                        x["demand_avg2week_sameweekday_samesnap0"],
                                                                        x["demand_avg2week_sameweekday_samesnap1"]))
      .assign(demand_avg4week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                        x["demand_avg4week_sameweekday_samesnap0"],
                                                                        x["demand_avg4week_sameweekday_samesnap1"]))
      .assign(demand_avg12week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                         x["demand_avg12week_sameweekday_samesnap0"],
                                                                         x["demand_avg12week_sameweekday_samesnap1"]))
      .assign(demand_avg48week_sameweekday_samesnap = lambda x: np.where(x["snap"] == 0,
                                                                         x["demand_avg48week_sameweekday_samesnap0"],
                                                                         x["demand_avg48week_sameweekday_samesnap1"]))
      .drop(columns = ['demand_avg2week_sameweekday_samesnap0', 'demand_avg2week_sameweekday_samesnap1',
                       'demand_avg4week_sameweekday_samesnap0', 'demand_avg4week_sameweekday_samesnap1',
                       'demand_avg12week_sameweekday_samesnap0', 'demand_avg12week_sameweekday_samesnap1',
                       'demand_avg48week_sameweekday_samesnap0', 'demand_avg48week_sameweekday_samesnap1'])
      .reset_index())
print(datetime.now() - tmp)
del df_tsfe
del df_tsfe_sameweekday
df.to_feather("df_final.ftr")


# Same analysis as above for metric
metr = df_meta_sub.query("h_dep == 'Y' and modeltype == 'metr'")["variable"].values
print(df[metr].dtypes)
print("\n\n misspct:\n", df[metr].isnull().mean().round(3).sort_values(ascending = False))
print("\n\n varimp_metr: \n", calc_imp(df.sample(n = int(1e5)), metr, target = "demand", target_type = "REGR"))
print("\n\n varimp_metr_fold: \n", calc_imp(df.sample(n = int(1e5)), metr, target = "fold_num"))

# No new categorical variables
print(df_meta_sub.query("h_dep == 'Y' and modeltype == 'cate'")["variable"].values)


########################################################################################################################
# Model
########################################################################################################################

#df = pd.read_feather("df_final.ftr")
df_train = df.query("fold == 'train'").reset_index(drop = True)
df_test = df.query("fold == 'test'").reset_index(drop = True)
del df
gc.collect()

# --- Define final features --------------------------------------------------------------------------------------------

metr = df_meta_sub.query("modeltype == 'metr'")["variable"].values
cate = df_meta_sub.query("modeltype == 'cate'")["variable"].values
all_features = np.concatenate([metr, cate])
setdiff(all_features, df_train.columns.values.tolist())
setdiff(df_train.columns.values.tolist(), all_features)


# --- Tune -------------------------------------------------------------------------------------------------------------

tune = False
if tune:
    # Sample
    n = 1e6
    df_tune = pd.concat([(df_train.query("myfold == 'train'")
                          .sample(n = int(n), random_state = 1)
                          .reset_index(drop = True)),
                         (df_train.query("myfold == 'test'"))]).reset_index(drop = True)

    # LightGBM
    start = time.time()
    fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor(n_jobs = n_jobs),
                             {"n_estimators": [x for x in range(1100, 3100, 1000)], "learning_rate": [0.02],
                              "num_leaves": [31], "min_child_samples": [10],
                              "colsample_bytree": [0.1], "subsample": [1], "subsample_freq": [1],
                              "objective": ["rmse"]},
                             cv = TrainTestSep(1, fold_var = "myfold").split(df_tune),
                             refit = False,
                             scoring = d_scoring["REGR"],
                             return_train_score = True,
                             n_jobs = 1)
           .fit(df_tune[all_features], df_tune["demand"], categorical_feature = cate.tolist()))
    print((time.time()-start)/60)
    pd.DataFrame(fit.cv_results_)
    plot_cvresult(fit.cv_results_, metric = "rmse",
                  x_var = "n_estimators", color_var = "min_child_samples", style_var = "learning_rate",
                  column_var = "objective", row_var = "colsample_bytree")


# --- Fit and Score ----------------------------------------------------------------------------------------------------

# Sample with weight
df_train = (df_train.sample(frac = 1, replace = True, weights = "weight_rmse", random_state = 1)
            .reset_index(drop = True))

# Fit
lgb_param = dict(n_estimators = 8000, learning_rate = 0.02,
                 num_leaves = 31, min_child_samples = 10,
                 colsample_bytree = 0.1, subsample = 1,
                 objective = "rmse",
                 n_jobs = n_jobs)
fit = (lgbm.LGBMRegressor(**lgb_param)
       .fit(X = df_train[all_features],
            y = df_train["demand"],
            #sample_weight = df_train["weight_all"].values/min(df_train["weight_all"]),
            categorical_feature = cate.tolist()))

# Score
yhat = fit.predict(df_test[all_features])
# rmse(yhat, df_test["demand"])
df_test["yhat"] = np.where((df_test["sell_price_isna"] == 1) | (yhat < 0), 0, yhat)
# rmse(df_test["yhat"], df_test["demand"])
#df_test[["yhat","yhat_w","demand"]].corr()
#rmse(df_test["yhat"], df_test["demand"])

# --- Write submission -------------------------------------------------------------------------------------------------

df_tmp = df_test[["id", "d", "yhat"]].set_index(["id", "d"]).unstack("d").reset_index()
df_submit = pd.concat([pd.DataFrame(df_tmp.iloc[:, 1:29].values.round(5)).assign(id = df_tmp["id"]),
                       (pd.DataFrame(df_tmp.iloc[:, 1:29].values.round(5)).assign(id = df_tmp["id"]
                                                                           .str.replace("validation", "evaluation")))])
df_submit.columns = ["F" + str(i) for i in range(1, 29)] + ["id"]
(pd.read_csv(dataloc + "sample_submission.csv")[["id"]]
 .merge(df_submit, on = "id", how = "left")
 .fillna(0)
 .to_csv("data/submit.csv", index = False))


'''
df_my = pd.read_csv("data/submit_firsttry.csv").iloc[:30490, :]
df_kernel = df_my[["id"]].merge(pd.read_csv("data/submission_kernel.csv").iloc[:30490, :], how="left")
df_t = df_my[["id"]].merge(pd.read_csv("data/sales_train_validation.csv").iloc[:30490, :], how="left")


df_tmp = pd.DataFrame(dict(my=df_my.mean(axis=1).values,
                           kernel=df_kernel.mean(axis=1).values,
                           train = df_t.iloc[:,6:].mean(axis=1).values))
df_tmp = pd.DataFrame(dict(my=df_my.iloc[:, 1:].values.flatten(),
                           kernel=df_kernel.iloc[:, 1:].values.flatten()))
df_tmp.corr()
plt.scatter(df_tmp.my, df_tmp.train, s=0.1)
plt.scatter(df_tmp.kernel, df_tmp.train, s=0.1)

df_blub = df.groupby(["id","myfold"])["demand"].mean().unstack("myfold")
plt.scatter(df_blub.train, df_blub.test, s=0.1)


'''







