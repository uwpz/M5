
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *

# Specific libraries
from sklearn.model_selection import GridSearchCV, cross_validate, ShuffleSplit, learning_curve
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm

# Main parameter
TARGET_TYPE = "CLASS"

# Specific parameters
if TARGET_TYPE == "CLASS":
    metric = "roc_auc" # metric for peformance comparison
if TARGET_TYPE == "REGR":
    metric = make_scorer(spearman_loss_func, greater_is_better=True)

# Load results from exploration
with open(TARGET_TYPE + "_1_explore.pkl", "rb") as file:
    d_vars = pickle.load(file)
df, metr, cate, features, features_binned, features_lgbm = \
    d_vars["df"], d_vars["metr"], d_vars["cate"], d_vars["features"], d_vars["features_binned"], d_vars["features_lgbm"]


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid)
# ######################################################################################################################

# --- Sample data ----------------------------------------------------------------------------------------------------
if TARGET_TYPE == "CLASS":
    # Undersample only training data (take all but n_maxpersample at most)
    under_samp = Undersample(n_max_per_level=500)
    df_tmp = under_samp.fit_transform(df)
    b_all = under_samp.b_all
    b_sample = under_samp.b_sample
    print(b_sample, b_all)
    # df_tmp, b_sample, b_all = undersample_n(df[df["fold"] == "train"], 500)
    df_tune = pd.concat([df_tmp, df.loc[df["fold"] == "test"]], sort=False)
    df_tune.groupby("fold")["target"].describe()

if TARGET_TYPE == "REGR":
    df_tune = df.sample(n=min(df.shape[0], int(5e3)))


# --- Define some splits -------------------------------------------------------------------------------------------
# split_index = PredefinedSplit(df_tune["fold"].map({"train": -1, "test": 0}).values)
split_my1fold_cv = TrainTestSep(1)
# split_5fold = KFold(5, shuffle=False, random_state=42)
split_my5fold_cv = TrainTestSep(5)
split_my5fold_boot = TrainTestSep(5, "bootstrap")
'''
df_tune["fold"].value_counts()
split_my5fold = TrainTestSep(n_splits=5, sample_type="cv")
iter_split = split_my5fold.split(df_tune)
i_train, i_test = next(iter_split)
df_tune["fold"].iloc[i_train].describe()
df_tune["fold"].iloc[i_test].describe()
i_test.sort()
i_test
'''


# --- Fits -----------------------------------------------------------------------------------------------------------
# Lasso / Elastic Net
fit = GridSearchCV(ElasticNet(normalize=True, warm_start=True),
                   [{"alpha": [2 ** x for x in range(-6, -15, -1)],
                     "l1_ratio": [0, 0.2, 0.5, 0.8, 1]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4)\
    .fit(CreateSparseMatrix(cate=features_binned, df_ref=df_tune).fit_transform(df_tune), df_tune["target"])
print(fit.best_params_)
pd.DataFrame.from_dict(fit.cv_results_)\
    .pivot_table(["mean_test_score"], index="param_alpha", columns="param_l1_ratio")\
    .plot(marker="o")
# -> keep l1_ratio=1 to have a full Lasso


# Random Forest
# noinspection PyTypeChecker
fit = GridSearchCV(RandomForestRegressor(warm_start=True) if TARGET_TYPE == "REGR" else RandomForestClassifier(warm_start=True),
                   [{"n_estimators": [10, 20],
                     "max_features": [x for x in range(1, len(features), 5)]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   # use_warm_start=["n_estimators"],
                   n_jobs=4)\
    .fit(CreateSparseMatrix(metr=metr, cate=cate, df_ref=df_tune).fit_transform(df_tune), df_tune["target"])
print(fit.best_params_)
pd.DataFrame.from_dict(fit.cv_results_)\
    .pivot_table(["mean_test_score"], index="param_n_estimators", columns="param_max_features")\
    .plot(marker="o")
# -> keep around the recommended values: max_features = floor(sqrt(length(features)))


# XGBoost
fit = GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                   [{"n_estimators": [x for x in range(100,3100,500)], "learning_rate": [0.01],
                     "max_depth": [3,6], "min_child_weight": [5,10]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   # use_warm_start="n_estimators",
                   n_jobs=4) \
    .fit(CreateSparseMatrix(metr=metr, cate=cate, df_ref=df_tune).fit_transform(df_tune), df_tune["target"])
print(fit.best_params_)
# -> keep around the recommended values: max_depth = 6, shrinkage = 0.01, n.minobsinnode = 10
df_fitres = pd.DataFrame.from_dict(fit.cv_results_)
df_fitres.mean_fit_time.values.mean()
sns.FacetGrid(df_fitres, col="param_min_child_weight", margin_titles=True) \
    .map(sns.lineplot, "param_n_estimators", "mean_test_score",  # do not specify x= and y=!
         hue="#" + df_fitres["param_max_depth"].astype('str'),  # needs to be string not starting with "_"
         style=df_fitres["param_learning_rate"],
         marker="o").add_legend()


# LightGBM
fit = GridSearchCV(lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else lgbm.LGBMClassifier(),
                   [{"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                     "num_leaves": [8,32,64], "min_child_samples": [5,10]}],
                   cv=split_my1fold_cv.split(df_tune),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4) \
    .fit(df_tune[features_lgbm], df_tune["target"],
         categorical_feature=[x for x in features_lgbm.tolist() if "_ENCODED" in x]
         )
print(fit.best_params_)
df_fitres = pd.DataFrame.from_dict(fit.cv_results_)
sns.FacetGrid(df_fitres, col="param_min_child_samples", margin_titles=True) \
    .map(sns.lineplot, "param_n_estimators", "mean_test_score",  # do not specify x= and y=!
         hue="#" + df_fitres["param_num_leaves"].astype('str'),  # needs to be string without not starting with "_"
         style=df_fitres["param_learning_rate"],
         marker="o").add_legend()


# ######################################################################################################################
# Evaluate generalization gap
# ######################################################################################################################

# Sample data (usually undersample training data)
df_gengap = df_tune.copy()


# Tune grid to loop over
param_grid = [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
               "max_depth": [3, 6], "min_child_weight": [5, 10],
               "colsample_bytree": [0.7], "subsample": [0.7],
               "gamma": [10]}]

# Calc generalization gap
fit = GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                   param_grid,
                   cv=split_my1fold_cv.split(df_gengap),
                   refit=False,
                   scoring=metric,
                   return_train_score=True,
                   n_jobs=4) \
    .fit(CreateSparseMatrix(metr=metr, cate=cate, df_ref=df_gengap).fit_transform(df_gengap), df_gengap["target"])
df_gengap_result = pd.DataFrame.from_dict(fit.cv_results_)\
    .rename(columns={"mean_test_score": "test",
                     "mean_train_score": "train"})\
    .assign(train_test_score_diff=lambda x: x.train - x.test)\
    .reset_index()


# Plot generalization gap
pdf_pages = PdfPages(plotloc + TARGET_TYPE + "_gengap.pdf")
sns.FacetGrid(df_gengap_result, col="param_min_child_weight", row="param_gamma",
              margin_titles=True, height=5) \
    .map(sns.lineplot, "param_n_estimators", "train_test_score_diff",
         hue="#" + df_gengap_result["param_max_depth"].astype('str'),
         marker="o").add_legend()
pdf_pages.savefig()
df_plot = pd.melt(df_gengap_result,
                  id_vars=np.setdiff1d(df_gengap_result.columns.values, ["test", "train"]),
                  value_vars=["test", "train"],
                  var_name="fold", value_name="score")
sns.FacetGrid(df_plot, col="param_min_child_weight", row="param_gamma",
              margin_titles=True, height=5) \
    .map(sns.lineplot, "param_n_estimators", "score",
         hue="#" + df_plot["param_max_depth"].astype('str'),
         style=df_plot["fold"],
         marker="o").add_legend()
pdf_pages.savefig()
pdf_pages.close()


# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_modelcomp = df_tune.copy()


# --- Run methods ------------------------------------------------------------------------------
df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
      estimator=GridSearchCV(ElasticNet(normalize=True, warm_start=True),
                             [{"alpha": [2 ** x for x in range(-6, -15, -1)],
                               "l1_ratio": [1]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                             refit=True,
                             scoring=metric,
                             return_train_score=False,
                             n_jobs=4),
      X=CreateSparseMatrix(cate=features_binned, df_ref=df_modelcomp).fit_transform(df_modelcomp),
      y=df_modelcomp["target"],
      cv=split_my5fold_cv.split(df_modelcomp),
      return_train_score=False,
      n_jobs=4)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="ElasticNet"),
                                                 ignore_index=True)

# Xgboost
cvresults = cross_validate(
      estimator=GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                             [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
                               "max_depth": [6], "min_child_weight": [10]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                             refit=True,
                             scoring=metric,
                             return_train_score=False,
                             n_jobs=4),
      X=CreateSparseMatrix(metr=metr, cate=cate, df_ref=df_modelcomp).fit_transform(df_modelcomp),
      y=df_modelcomp["target"],
      cv=split_my5fold_cv.split(df_modelcomp),
      return_train_score=False,
      n_jobs=4)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model="XGBoost"),
                                                 ignore_index=True)


# --- Plot model comparison ------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 1)
sns.boxplot(data=df_modelcomp_result, x="model", y="test_score", ax=ax)
sns.lineplot(data=df_modelcomp_result, x="model", y="test_score",
             hue="#" + df_modelcomp_result["index"].astype("str"), linewidth=0.5, linestyle=":",
             legend=None, ax=ax)
fig.savefig(plotloc + TARGET_TYPE + "_model_comparison.pdf")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################

# Basic data sampling
df_lc = df

# Calc learning curve
n_train, score_train, score_test = learning_curve(
      estimator=GridSearchCV(xgb.XGBRegressor() if TARGET_TYPE == "REGR" else xgb.XGBClassifier(),
                             [{"n_estimators": [x for x in range(100, 1100, 200)], "learning_rate": [0.01],
                               "max_depth": [3], "min_child_weight": [10]}],
                             cv=ShuffleSplit(1, 0.2, random_state=999),  # just 1-fold for tuning
                             refit=True,
                             scoring=metric,
                             return_train_score=False,
                             n_jobs=4),
      X=CreateSparseMatrix(metr=metr, cate=cate, df_ref=df_lc).fit_transform(df_lc),
      y=df_lc["target"],
      train_sizes=np.append(np.linspace(0.05, 0.1, 5), np.linspace(0.2, 1, 5)),
      cv=split_my1fold_cv.split(df_lc),
      n_jobs=4)
df_lc_result = pd.DataFrame(zip(n_train, score_train[:, 0], score_test[:, 0]),
                            columns=["n_train", "train", "test"])\
    .melt(id_vars="n_train", value_vars=["train", "test"], var_name="fold", value_name="score")

# Plot learning curve
fig, ax = plt.subplots(1, 1)
sns.lineplot(x="n_train", y="score", hue="fold", data=df_lc_result, marker="o", ax=ax)
fig.savefig(plotloc + TARGET_TYPE + "_learningCurve.pdf")


plt.close("all")
