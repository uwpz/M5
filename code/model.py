
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
horizon = 14
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

# ---  Join depending on horizon ---------------------------------------------------------------------------------------
#df = df.query("year >= 2014").reset_index(drop = True) #TODO

tmp = datetime.now()
df = (df.set_index(["date", "id"])
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
del df_tsfe
del df_tsfe_sameweekday
df = reduce_mem_usage(df, float_convert = True)


# Same analysis as above for metric
metr = df_meta_sub.query("h_dep == 'Y' and modeltype == 'metr'")["variable"].values
print(df[metr].dtypes)
print("\n\n misspct:\n", df[metr].isnull().mean().round(3).sort_values(ascending = False))
print("\n\n varimp_metr: \n", calc_imp(df.sample(n = int(1e5)), metr, target = "demand", target_type = "REGR"))
print("\n\n varimp_metr_fold: \n", calc_imp(df.sample(n = int(1e5)), metr, target = "fold_num"))

# No new categorical variables
print(df_meta_sub.query("h_dep == 'Y' and modeltype == 'cate'")["variable"].values)

# Check
print(setdiff(df.columns.values, df_meta_sub["variable"].values))
print(setdiff(df_meta_sub["variable"].values, df.columns.values))


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
#df = df.query("myfold != 'util'").reset_index(drop = True)
#df["demand_normed"] = (df["demand"] - df["demand_median"]) / np.where(df["demand_iqr"] == 0, 1, df["demand_iqr"]) #TODO
#df["demand_normed"] = (df["demand"] - df["demand_mean"]) / np.where(df["demand_sd"] == 0, 1, df["demand_sd"]) #TODO
#df["demand_normed"] = (df["demand"] - df["demand_median"])  #TODO
#df["demand_normed"] = np.where(df["demand_median"] == 0, df["demand"], df["demand"]/df["demand_median"])  #TODO

df_train = df.query("fold == 'train'").reset_index(drop = True)  # .query("year >= 2014")
df_test = df.query("fold == 'test'").reset_index(drop = True)
del df  # TODO
gc.collect()

metr = df_meta_sub.query("modeltype == 'metr'")["variable"].values
#metr = metr[~ (pd.Series(metr).str.contains("max") | pd.Series(metr).str.contains("min"))]
#metr = np.append(metr, ["demand_median","demand_iqr"])
cate = df_meta_sub.query("modeltype == 'cate'")["variable"].values
all_features = np.concatenate([metr, cate])
setdiff(all_features, df_train.columns.values.tolist())
setdiff(df_train.columns.values.tolist(), all_features)
#all_features = setdiff(all_features, "id_copy") #TODO
#cate = setdiff(cate, ["id_copy","item_id"]) #TODO
#cate = np.array(["dayofweek","id_copy","item_id"])#TODO

#cate = np.array(["dayofweek"])
print(datetime.now() - begin)


# --- Tune -------------------------------------------------------------------------------------------------------------

tune = False
if tune:

    # Check: >= 2014, remove year

    # Sample
    n = 5e6
    df_tune = pd.concat([(df_train.query("myfold == 'train'")
                          .assign(weight_sales = lambda x: x["weight_sales"].pow(0.5))
                          #.query("year >= 2014")
                          #.sample(n = int(n), random_state = 1)
                          .sample(frac = 1, replace = True, weights = "weight_sales", random_state = 2)),
                         (df_train.query("myfold == 'test'"))]).reset_index(drop = True)
    #df_tune = df_tune.query("anydemand == 1").reset_index()


    def wrmsse(y_true, y_pred):
        #pdb.set_trace()
        df_holdout = df_tune.iloc[y_true.index.values]
        df_rmse = pd.DataFrame()
        for key in d_comb:
            df_tmp = (df_holdout.assign(yhat = y_pred)
                      #.assign(yhat = lambda x: (x["yhat"] * np.where(x["demand_iqr"] == 0, 1, x["demand_iqr"])) + x["demand_median"]) #TODO
                      #.assign(yhat = lambda x: (x["yhat"] * np.where(x["demand_sd"] == 0, 1, x["demand_sd"])) + x["demand_mean"]) #TODO
                      #.assign(yhat = lambda x: x["yhat"] + x["demand_median"]) #TODO
                      #.assign(yhat = lambda x: np.where(x["demand_median"] == 0, x["yhat"], x["yhat"] * x["demand_median"])) #TODO
                      .assign(dummy = "dummy")
                      .groupby(d_comb[key] + ["date"])["demand", "yhat"].sum().reset_index("date", drop = True)
                      .groupby(d_comb[key]).apply(lambda x: pd.Series({"rmse": rmse(x["demand"], x["yhat"])}))
                      .assign(key = key)
                      .reset_index())
            df_rmse = pd.concat([df_rmse, df_tmp], ignore_index = True)
        return (df_rmse.merge(df_help, how = "left")
                  .eval("wrmsse = sales * rmse/rmse_denom")["wrmsse"].sum())

    # LightGBM
    start = time.time()
    fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor(n_jobs = n_jobs),
                             {"n_estimators": [x for x in range(500, 3500, 500)], "learning_rate": [0.04],
                              "num_leaves": [63], "min_child_samples": [10],
                              "colsample_bytree": [0.6], "subsample": [1], "subsample_freq": [1],
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
                  x_var = "n_estimators", color_var = "learning_rate", style_var = "min_child_samples",
                  column_var = "subsample", row_var = "colsample_bytree")
    plot_cvresult(fit.cv_results_, metric = "wrmsse",
                  x_var = "n_estimators", color_var = "learning_rate", style_var = "min_child_samples",
                  column_var = "subsample", row_var = "colsample_bytree")


# --- Fit and Score ----------------------------------------------------------------------------------------------------

# Sample with weight
df_train = (df_train
            .assign(weight_sales = lambda x: x["weight_sales"].pow(0.5))
            .sample(frac = 1, replace = True, weights = "weight_sales", random_state = 2)
            .reset_index(drop = True))
#df_train_reg = df_train.query("anydemand == 1").reset_index(drop = True)  # TODO


# Fit
lgb_param = dict(n_estimators = 3000, learning_rate = 0.04,  # TODO
                 num_leaves = 63, min_child_samples = 10,
                 colsample_bytree = 0.6, subsample = 1,
                 #objective = "rmse",
                 n_jobs = n_jobs)
fit = (lgbm.LGBMRegressor(**lgb_param)
       .fit(X = df_train[all_features],
            y = df_train["demand"],
            sample_weight = df_train["weight_rmse"].values / min(df_train["weight_rmse"]),
            categorical_feature = cate.tolist()))

# fit_reg = (lgbm.LGBMRegressor(**lgb_param) # TODO
#        .fit(X = df_train_reg[all_features], # TODO
#             y = df_train_reg["demand"],  # TODO
#             #sample_weight = df_train["weight_rmse"].values,#/min(df_train["weight_rmse"]),
#             categorical_feature = cate.tolist()))
# fit_class = (lgbm.LGBMClassifier(**lgb_param) # TODO
#        .fit(X = df_train[all_features], # TODO
#             y = df_train["anydemand"],  # TODO
#             categorical_feature = cate.tolist()))

# Score

df_test["yhat"] = fit.predict(df_test[all_features])
#df_test["yhat"] = fit_class.predict_proba(df_test[all_features])[:,1] * fit_reg.predict(df_test[all_features]) # TODO
#df_test["yhat"] = (df_test["yhat"] * np.where(df_test["demand_iqr"] == 0, 1, df_test["demand_iqr"])
#                   + df_test["demand_median"])  #TODO
df_test["yhat"] = np.where((df_test["sell_price_isna"] == 1) | (df_test["yhat"] < 0), 0.0001, df_test["yhat"] )

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
                        #.assign(id = df_tmp["id"].str.replace("validation", "evaluation")))])
                        .assign(id = df_tmp["id"].str.replace("evaluation", "validation")))])
df_submit.columns = ["F" + str(i) for i in range(1, 29)] + ["id"]
(pd.read_csv(dataloc + "sample_submission.csv")[["id"]]
 .merge(df_submit, on = "id", how = "left")
 .fillna(0)
 .to_csv("data/submit.csv", index = False))


'''
df_tmp = pd.read_csv("data/submit_secondtry.csv")
df_tmp.iloc[:, 1:] = np.round(df_tmp.iloc[:, 1:]*1, 5)
df_tmp .to_csv("data/submit.csv", index = False)
'''


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






