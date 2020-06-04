
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
begin = datetime.now()

# Specific parameter
n_sample = 5000
n_jobs = 16
horizon = 28
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
tmp = TargetEncoding(features = cate, encode_flag_column = "encode_flag", target = "demand",
                    remove_burned_data = False,
                    suffix = "").fit(df)
df = tmp.transform(df)

# Adapt df_help with encoding
help_columns = ["cat_id","dept_id","item_id","state_id","store_id"]
df_help[help_columns] = df_help[help_columns].apply(lambda x: x.map(tmp._d_map[x.name]))
                                             #       .fillna(np.median(list(tmp._d_map[x.name].values()))))




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
'''

# --- Eval metric help dataframes -------------------------------------------------------------------------------------

# Aggregation levels
d_comb = {1: ["dummy"],
          2: ["state_id"], 3: ["store_id"], 4: ["cat_id"], 5: ["dept_id"],
          6: ["state_id", "cat_id"], 7: ["state_id", "dept_id"], 8: ["store_id", "cat_id"], 9: ["store_id", "dept_id"],
          10: ["item_id"], 11: ["item_id", "state_id"], 12: ["item_id", "store_id"]}

# Sales
df_sales = pd.DataFrame()
denom = 12 * df.query("myfold == 'test'")["sales"].sum()
for key in d_comb:
    df_tmp = (df.query("myfold == 'test'").assign(dummy = "dummy").groupby(d_comb[key])[["sales"]].sum()
              .assign(sales = lambda x: x["sales"]/denom)
              .assign(key = key)
              .reset_index())
    df_sales = pd.concat([df_sales, df_tmp], ignore_index = True)

# rmse_denom
df_rmse_denom = pd.DataFrame()
for key in d_comb:
    df_tmp = (df.query("fold == 'train'").assign(dummy = "dummy")
              .groupby(d_comb[key] + ["date"])["demand", "lagdemand"].sum().reset_index("date", drop = True)
              .groupby(d_comb[key]).apply(lambda x: pd.Series({"rmse_denom": rmse(x["demand"], x["lagdemand"])}))
              .assign(key = key)
              .reset_index())
    df_rmse_denom = pd.concat([df_rmse_denom, df_tmp], ignore_index = True)

'''

# --- Prepare data and define final features ---------------------------------------------------------------------------
#df.to_feather("df_final.ftr")
#df = pd.read_feather("df_final.ftr")
df = df.query("myfold != 'util'").reset_index(drop = True)
df_train = df.query("myfold == 'train'").reset_index(drop = True)  # TODO
df_test = df.query("myfold == 'test'").reset_index(drop = True)
del df  # TODO
gc.collect()

metr = df_meta_sub.query("modeltype == 'metr'")["variable"].values
cate = df_meta_sub.query("modeltype == 'cate'")["variable"].values
all_features = np.concatenate([metr, cate])
setdiff(all_features, df_train.columns.values.tolist())
setdiff(df_train.columns.values.tolist(), all_features)

print(datetime.now() - begin)


# --- Tune -------------------------------------------------------------------------------------------------------------

tune = False
if tune:

    # Check: >= 2014, remove year

    # Sample
    n = 10e6
    df_tune = pd.concat([(df_train.query("myfold == 'train'")
                          #.assign(weight_sales = lambda x: x["weight_sales"].pow(0.5))
                          #.query("year >= 2014")
                          .sample(n = int(n), random_state = 1)
                          .sample(frac = 1, replace = True, weights = "weight_sales", random_state = 2)),
                         (df_train.query("myfold == 'test'"))]).reset_index(drop = True)


    def wrmsse(y_true, y_pred):
        #pdb.set_trace()
        df_holdout = df_tune.iloc[y_true.index.values]
        df_rmse = pd.DataFrame()
        for key in d_comb:
            df_tmp = (df_holdout.assign(yhat = y_pred)
                      .assign(dummy = "dummy")
                      .groupby(d_comb[key] + ["date"])["demand", "yhat"].sum().reset_index("date", drop = True)
                      .groupby(d_comb[key]).apply(lambda x: pd.Series({"rmse": rmse(x["demand"], x["yhat"])}))
                      .assign(key = key)
                      .reset_index())
            df_rmse = pd.concat([df_rmse, df_tmp], ignore_index = True)
        #return (df_rmse.merge(df_rmse_denom, how = "left").merge(df_sales, how = "left")
        return (df_rmse.merge(df_help, how = "left")
                  .eval("wrmsse = sales * rmse/rmse_denom")["wrmsse"].sum())

    # LightGBM
    start = time.time()
    fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor(n_jobs = n_jobs),
                             {"n_estimators": [x for x in range(600, 4600, 200)], "learning_rate": [0.01],
                              "num_leaves": [31], "min_child_samples": [10],
                              "colsample_bytree": [0.1], "subsample": [1], "subsample_freq": [1],
                              "objective": ["rmse"]},
                             cv = TrainTestSep(1, fold_var = "myfold").split(df_tune),
                             refit = False,
                             #scoring = d_scoring["REGR"],
                             scoring = {"rmse": make_scorer(rmse, greater_is_better = False),
                                        "wrmsse": make_scorer(wrmsse, greater_is_better = False)},
                             return_train_score = False,
                             n_jobs = 1)
           .fit(df_tune[all_features], df_tune["demand"], categorical_feature = cate.tolist()))
    print((time.time()-start)/60)
    pd.DataFrame(fit.cv_results_)
    plot_cvresult(fit.cv_results_, metric = "rmse",
                  x_var = "n_estimators", color_var = "min_child_samples", style_var = "colsample_bytree",
                  column_var = "num_leaves", row_var = "learning_rate")
    plot_cvresult(fit.cv_results_, metric = "wrmsse",
                  x_var = "n_estimators", color_var = "min_child_samples", style_var = "colsample_bytree",
                  column_var = "num_leaves", row_var = "learning_rate")


# --- Fit and Score ----------------------------------------------------------------------------------------------------

# Sample with weight
# df_train = (df_train.sample(frac = 1, replace = True, weights = "weight_all", random_state = 2)  # TODO
#             #.query("year >= 2014")
#             .reset_index(drop = True))

# Fit
lgb_param = dict(n_estimators = 1000, learning_rate = 0.04,  # TODO
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
df_test["yhat"] = np.where((df_test["sell_price_isna"] == 1) | (yhat < 0), 0, yhat)

'''
# Rmse
rmse(df_test["yhat"], df_test["demand"])

df_rmse = pd.DataFrame()
for key in d_comb:
    df_tmp = (df_test.assign(dummy = "dummy")
              .groupby(d_comb[key] + ["date"])["demand", "yhat"].sum().reset_index("date", drop = True)
              .groupby(d_comb[key]).apply(lambda x: pd.Series({"rmse": rmse(x["demand"], x["yhat"])}))
              .assign(key = key)
              .reset_index())
    df_rmse = pd.concat([df_rmse, df_tmp], ignore_index = True)
df_tmp = df_rmse.merge(df_help, how = "left").eval("wrmsse = sales * rmse/rmse_denom")
df_tmp.groupby("key")["wrmsse"].sum()
df_tmp["wrmsse"].sum()
'''

'''

df_check = (df_test.groupby("id")["yhat", "demand"].mean()
            .join(df_train.groupby("id")[["demand"]].mean().rename(columns = {"demand": "demand_train"}))
            .eval("yhat_minus_demand = yhat-demand")
            .sort_values("yhat_minus_demand")
            .reset_index())
'''

# --- Write submission -------------------------------------------------------------------------------------------------

df_tmp = df_test[["id", "d", "yhat"]].set_index(["id", "d"]).unstack("d").reset_index()
df_submit = pd.concat([pd.DataFrame(df_tmp.iloc[:, 1:29].values.round(5)).assign(id = df_tmp["id"]),
                       (pd.DataFrame(df_tmp.iloc[:, 1:29].values.round(5))
                        .assign(id = df_tmp["id"].str.replace("validation", "evaluation")))])
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







