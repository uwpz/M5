# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# sys.path.append(os.getcwd() + "\\code")  # not needed if code is marked as "source" in pycharm

# Specific libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # , GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression  # , ElasticNet
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
plt.ion(); matplotlib.use('TkAgg')
#  from sklearn.tree import DecisionTreeRegressor, plot_tree , export_graphviz

# Main parameter
horizon = 28
TARGET_TYPE = "CLASS"

# Specific parameters
n_jobs = 4
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    metric = "auc"  # metric for peformance comparison
else:
    metric = "pear"

# Load results from exploration
df = metr_standard = cate_standard = metr_binned = cate_binned = metr_encoded = cate_encoded = target_labels = None
with open("1_explore_h" + str(horizon) + ".pkl", "rb") as file:
    d_pick = pickle.load(file)
for key, val in d_pick.items():
    exec(key + "= val")

# Scale "metr_enocded" features for DL (Tree-based are not influenced by this Trafo)
df[metr_encoded] = (df[metr_encoded] - df[metr_encoded].min()) / (df[metr_encoded].max() - df[metr_encoded].min())
df[metr_encoded].describe()
df["year_ENCODED"] = df["year"].astype("int")


# ######################################################################################################################
# # Test an algorithm (and determine parameter grid)
# ######################################################################################################################

# --- Sample data ----------------------------------------------------------------------------------------------------
df["myfold"].value_counts()
df["anydemand"].mean()
n = 50e3
if TARGET_TYPE in ["CLASS", "MULTICLASS"]:
    df_tune = pd.concat([(df.query("myfold == 'train'")
                          .sample(n = int(n), random_state = 1).reset_index(drop = True)),
                         df.query("myfold == 'test'")]).reset_index(drop = True)
    df_tune["target"] = df_tune["anydemand"]
    '''
    # Undersample only training data (take all but n_maxpersample at most)
    under_samp = Undersample(n_max_per_level = 5000000)
    df_tmp = under_samp.fit_transform(df.query("myfold == 'train'").reset_index())
    b_all = under_samp.b_all
    b_sample = under_samp.b_sample
    print(b_sample, b_all)
    df_tune = pd.concat([df_tmp, df.query("myfold == 'test'").reset_index(drop = True)], sort = False).reset_index(
        drop = True)
    df_tune.groupby("myfold")["target"].describe()
    '''
else:  # "REGR"
    df_tune = pd.concat([(df
                          #.query("anydemand == 1")
                          .query("myfold == 'train'")
                          .sample(n = int(n), random_state = 1).reset_index(drop = True)),
                         (df
                          #.query("anydemand == 1")
                          .query("myfold == 'test'"))]).reset_index(drop = True)
    df_tune["target"] = df_tune["demand"]
df_tune["myfold"].value_counts()
df_tune["anydemand"].mean()

# --- Define some splits -------------------------------------------------------------------------------------------

# split_index = PredefinedSplit(df_tune["myfold"].map({"train": -1, "test": 0}).values)
split_my1fold_cv = TrainTestSep(1, fold_var = "myfold")
# split_5fold = KFold(5, shuffle=False, random_state=42)
split_my5fold_cv = TrainTestSep(5, fold_var = "myfold")
split_my5fold_boot = TrainTestSep(5, "bootstrap", fold_var = "myfold")
'''
df_tune["myfold"].value_counts()
mysplit = split_my5fold_cv.split(df_tune)
i_train, i_test = next(mysplit)
df_tune["myfold"].iloc[i_train].describe()
df_tune["myfold"].iloc[i_test].describe()
i_test.sort()
i_test
'''


# --- Fits -----------------------------------------------------------------------------------------------------------

# Lasso / Elastic Net
fit = (GridSearchCV(SGDRegressor(penalty = "ElasticNet", warm_start = True) if TARGET_TYPE == "REGR" else
                    SGDClassifier(loss = "log", penalty = "ElasticNet", warm_start = True),  # , tol=1e-2
                    {"alpha": [2 ** x for x in range(-5, -12, -1)],
                     "l1_ratio": [1]},
                    cv = split_my1fold_cv.split(df_tune),
                    refit = False,
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = True,
                    n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr_binned, cate = cate_binned, df_ref = df_tune).fit_transform(df_tune),
            df_tune["target"]))
plot_cvresult(fit.cv_results_, metric = metric, x_var = "alpha", color_var = "l1_ratio")
pd.DataFrame(fit.cv_results_)
# -> keep l1_ratio=1 to have a full Lasso

# XGBoost
start = time.time()
fit = (GridSearchCV_xlgb(xgb.XGBRegressor(verbosity = 0) if TARGET_TYPE == "REGR" else xgb.XGBClassifier(verbosity = 0),
                         {"n_estimators": [x for x in range(100, 5100, 500)], "learning_rate": [0.01],
                          "max_depth": [3, 6], "min_child_weight": [10], "subsample": [0.01, 1]},
                         cv = split_my1fold_cv.split(df_tune),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_tune).fit_transform(df_tune),
            df_tune["target"]))
print(time.time()-start)
pd.DataFrame(fit.cv_results_)
plot_cvresult(fit.cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "max_depth", column_var = "subsample")


# -> keep around the recommended values: max_depth = 6, shrinkage = 0.01, n.minobsinnode = 10

# LightGBM
start = time.time()
fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else lgbm.LGBMClassifier(),
                         {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
                          "num_leaves": [8, 64], "min_child_samples": [10],
                          "colsample_bytree": [1], "bagging_fraction": [0.01,1],
                          "objective": ["poisson"]},
                         cv = split_my1fold_cv.split(df_tune),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(df_tune[metr_encoded], df_tune["target"],
            categorical_feature = [x for x in metr_encoded.tolist() if "_ENCODED" in x]))
print(time.time()-start)
plot_cvresult(fit.cv_results_, metric = metric,
              x_var = "n_estimators", color_var = "num_leaves", style_var = "min_child_samples", column_var = "bagging_fraction", row_var = "colsample_bytree")


# DeepL

# Keras wrapper for Scikit
def keras_model(input_dim, output_dim, target_type,
                size = "10",
                lambdah = None, dropout = None,
                lr = 1e-5,
                batch_normalization = False,
                activation = "relu"):
    model = Sequential()

    # Add dense layers
    for units in size.split("-"):
        model.add(Dense(units = int(units), activation = activation, input_dim = input_dim,
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None,
                        kernel_initializer = "glorot_uniform"))
        # Add additional layer
        if batch_normalization is not None:
            model.add(BatchNormalization())
        if dropout is not None:
            model.add(Dropout(dropout))

    # Output
    if target_type == "CLASS":
        model.add(Dense(1, activation = 'sigmoid',
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None))
        model.compile(loss = "binary_crossentropy", optimizer = optimizers.RMSprop(lr = lr), metrics = ["accuracy"])
    elif target_type == "MULTICLASS":
        model.add(Dense(output_dim, activation = 'softmax',
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None))
        model.compile(loss = "categorical_crossentropy", optimizer = optimizers.RMSprop(lr = lr),
                      metrics = ["accuracy"])
    else:
        model.add(Dense(1, activation = 'linear',
                        kernel_regularizer = l2(lambdah) if lambdah is not None else None))
        model.compile(loss = "mean_squared_error", optimizer = optimizers.RMSprop(lr = lr),
                      metrics = ["mean_squared_error"])

    return model


# Fit
fit = (GridSearchCV(KerasRegressor(build_fn = keras_model,
                                   input_dim = metr_encoded.size,
                                   output_dim = 1,
                                   target_type = TARGET_TYPE,
                                   verbose = 0) if TARGET_TYPE == "REGR" else
                    KerasClassifier(build_fn = keras_model,
                                    input_dim = metr_encoded.size,
                                    output_dim = 1 if TARGET_TYPE == "CLASS" else target_labels.size,
                                    target_type = TARGET_TYPE,
                                    verbose = 0),
                    {"size": ["10"],
                     "lambdah": [1e-8], "dropout": [None],
                     "batch_size": [40], "lr": [1e-3],
                     "batch_normalization": [True],
                     "activation": ["relu", "elu"],
                     "epochs": [2, 5, 10, 15]},
                    cv = split_my1fold_cv.split(df_tune),
                    refit = False,
                    scoring = d_scoring[TARGET_TYPE],
                    return_train_score = False,
                    n_jobs = 1)
       .fit(CreateSparseMatrix(metr = metr_encoded, df_ref = df_tune).fit_transform(df_tune),
            pd.get_dummies(df_tune["target"]) if TARGET_TYPE == "MULTICLASS" else df_tune["target"]))
plot_cvresult(fit.cv_results_, metric = metric, x_var = "epochs", color_var = "lambdah",
              column_var = "activation", row_var = "size")


# ######################################################################################################################
# Evaluate generalization gap
# ######################################################################################################################

# Sample data (usually undersample training data)
df_gengap = df_tune.copy()

# Tune grid to loop over
param_grid = {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
              "num_leaves": [16, 64], "min_child_samples": [10],
              "colsample_bytree": [1]}

# Calc generalization gap
fit = (GridSearchCV_xlgb(lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else lgbm.LGBMClassifier(),
                         param_grid,
                         cv = split_my1fold_cv.split(df_gengap),
                         refit = False,
                         scoring = d_scoring[TARGET_TYPE],
                         return_train_score = True,
                         n_jobs = n_jobs)
       .fit(df_gengap[metr_encoded], df_gengap["target"],
            categorical_feature = [x for x in metr_encoded.tolist() if "_ENCODED" in x]))
plot_gengap(fit.cv_results_, metric = metric,
            x_var = "n_estimators", color_var = "num_leaves", column_var = "min_child_samples",
            pdf = plotloc + TARGET_TYPE + "_lightgbm_gengap.pdf")


# ######################################################################################################################
# Simulation: compare algorithms
# ######################################################################################################################

# Basic data sampling
df_modelcomp = df_tune.copy()


# --- Run methods ------------------------------------------------------------------------------------------------------

df_modelcomp_result = pd.DataFrame()  # intialize

# Elastic Net
cvresults = cross_validate(
    estimator = GridSearchCV(SGDRegressor(penalty = "ElasticNet", warm_start = True) if TARGET_TYPE == "REGR" else
                             SGDClassifier(loss = "log", penalty = "ElasticNet", warm_start = True),  # , tol=1e-2
                             {"alpha": [2 ** x for x in range(-4, -12, -1)],
                              "l1_ratio": [1]},
                             cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
                             refit = metric,
                             scoring = d_scoring[TARGET_TYPE],
                             return_train_score = False,
                             n_jobs = n_jobs),
    X = CreateSparseMatrix(metr = metr_binned, cate = cate_binned, df_ref = df_modelcomp).fit_transform(df_modelcomp),
    y = df_modelcomp["target"],
    cv = split_my5fold_cv.split(df_modelcomp),
    return_train_score = False,
    n_jobs = n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model = "ElasticNet"),
                                                 ignore_index = True)

# LightGBM
cvresults = cross_validate(
    estimator = GridSearchCV_xlgb(
        lgbm.LGBMRegressor() if TARGET_TYPE == "REGR" else lgbm.LGBMClassifier(),
        {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
         "num_leaves": [16, 64], "min_child_samples": [10],
         "colsample_bytree": [1]},
        cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
        refit = metric,
        scoring = d_scoring[TARGET_TYPE],
        return_train_score = False,
        n_jobs = n_jobs),
    X = df_modelcomp[metr_encoded],
    y = df_modelcomp["target"],
    fit_params = dict(categorical_feature = [x for x in metr_encoded.tolist() if "_ENCODED" in x]),
    cv = split_my5fold_cv.split(df_modelcomp),
    return_train_score = False,
    n_jobs = n_jobs)
df_modelcomp_result = df_modelcomp_result.append(pd.DataFrame.from_dict(cvresults).reset_index()
                                                 .assign(model = "Lightgbm"),
                                                 ignore_index = True)


# --- Plot model comparison ------------------------------------------------------------------------------

plot_modelcomp(df_modelcomp_result.rename(columns = {"index": "run", "test_score": metric}), scorevar = metric,
               pdf = plotloc + TARGET_TYPE + "_model_comparison.pdf")


# ######################################################################################################################
# Learning curve for winner algorithm
# ######################################################################################################################

# Basic data sampling
df_lc = df_tune.copy()

# Calc learning curve
n_train, score_train, score_test = learning_curve(
    estimator = GridSearchCV_xlgb(
        xgb.XGBRegressor(verbosity = 0, objective = "count:poisson") if TARGET_TYPE == "REGR" else xgb.XGBClassifier(verbosity = 0),
        {"n_estimators": [x for x in range(100, 3100, 500)], "learning_rate": [0.01],
         "max_depth": [3], "min_child_weight": [10]},
        cv = ShuffleSplit(1, 0.2, random_state = 999),  # just 1-fold for tuning
        refit = metric,
        scoring = d_scoring[TARGET_TYPE],
        return_train_score = False,
        n_jobs = 4),
    X = CreateSparseMatrix(metr = metr_standard, cate = cate_standard, df_ref = df_lc).fit_transform(df_lc),
    y = df_lc["target"],
    train_sizes = np.append(np.linspace(0.05, 0.1, 5), np.linspace(0.2, 1, 5)),
    cv = split_my1fold_cv.split(df_lc),
    n_jobs = 4)

# Plot it
plot_learning_curve(n_train, score_train, score_test,
                    pdf = plotloc + TARGET_TYPE + "_learningCurve.pdf")

plt.close("all")
