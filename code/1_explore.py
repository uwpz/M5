
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from scipy.stats.mstats import winsorize


# Main parameter
horizon = 1

# Specific parameters
ylim = (-2, 2)
cutoff_corr = 0.4
cutoff_varimp = 0.1
color = None

# Load results from exploration
with open("0_etl_h" + str(horizon) + ".pkl", "rb") as file:
    d_vars = pickle.load(file)
df, df_meta_sub = d_vars["df"], d_vars["df_meta_sub"]


# ######################################################################################################################
# ETL
# ######################################################################################################################

# Train/Test fold: usually split by time
# np.random.seed(123)
# df["fold"] = np.random.permutation(pd.qcut(np.arange(len(df)), q=[0, 0.1, 0.8, 1], labels=["util", "train", "test"]))
# print(df.fold.value_counts())
# df["fold_num"] = df["fold"].map({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data
# df["encode_flag"] = df["fold"].map({"train": 0, "test": 0, "util": 1})  # Used for encoding


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------
metr = df_meta_sub.loc[df_meta_sub["type"] == "metr", "variable"].values
#df = Convert(features=metr, convert_to="float").fit_transform(df)
df[metr].describe()

# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
df[metr + "_BINNED_"] = df[metr].apply(lambda x: char_bins(x))

# Convert missings to own level ("(Missing)")
df[metr + "_BINNED_"] = df[metr + "_BINNED_"].replace("nan", np.nan).fillna("(Missing)")
print(create_values_df(df[metr + "_BINNED_"], 11))

# Get binned variables with just 1 bin (removed later)
onebin = (metr + "_BINNED_")[df[metr + "_BINNED_"].nunique() == 1]


# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------
# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
misspct.sort_values(ascending=False)  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
print(remove)
metr = setdiff(metr, remove)  # adapt metadata


# # Check for outliers and skewness
# df[metr].describe()
# plot_distr(df.query("fold != 'util'"), metr, target_type="REGR", color=color, ylim=ylim,
#            ncol=4, nrow=2, w=18, h=12, pdf=plotloc + "distr_metr_h" + str(horizon) + ".pdf")
#
# # Winsorize
# df[metr] = df[metr].apply(lambda x: x.clip(x.quantile(0.01),
#                                            x.quantile(0.99)))  # hint: plot again before deciding for log-trafo
#
# # Log-Transform
# tolog = np.array(["xxx"], dtype="object")
# df[tolog + "_LOG_"] = df[tolog].apply(lambda x: np.log(x - min(0, np.min(x)) + 1))
# metr = np.where(np.isin(metr, tolog), metr + "_LOG_", metr)  # adapt metadata (keep order)
# df.rename(columns=dict(zip(tolog + "_BINNED_", tolog + "_LOG_" + "_BINNED_")), inplace=True)  # adapt binned version


# --- Final variable information ------------------------------------------------------------------------------------
# Univariate variable importance
varimp_metr = calc_imp(df.query("fold != 'util'"), metr, target_type="REGR")
print(varimp_metr)
varimp_metr_binned = calc_imp(df.query("fold != 'util'"), metr + "_BINNED_", target_type="REGR")
print(varimp_metr_binned)

# Plot
plot_distr(df.query("fold != 'util'"), features=np.hstack(zip(metr, metr + "_BINNED_")),
           varimp=pd.concat([varimp_metr, varimp_metr_binned]), target_type="REGR", color=color, ylim=ylim,
           ncol=4, nrow=2, w=18, h=12, pdf=plotloc + "distr_metr_final_h" + str(horizon) + ".pdf")



# --- Removing variables -------------------------------------------------------------------------------------------
# Remove leakage features
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
plot_corr(df, metr, cutoff=cutoff_corr, pdf=plotloc + "corr_metr_h" + str(horizon) + ".pdf")
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)


# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------
miss = metr[df[metr].isnull().any().values]  # alternative: [x for x in metr if df[x].isnull().any()]

# Impute missings with randomly sampled value (or median, see below)
df = DfSimpleImputer(features=miss, strategy="median").fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------
# Nominal variables
cate = df_meta_sub.loc[df_meta_sub.type.isin(["cate"]), "variable"].values
df[cate] = df[cate].astype("str").replace("nan", np.nan)
df[cate].describe()


# --- Handling factor values ----------------------------------------------------------------------------------------
# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 50
levinfo = df[cate].apply(lambda x: x.unique().size).sort_values(ascending=False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

#
# # Create encoded features (for tree based models), i.e. numeric representation
# if TARGET_TYPE in ["REGR", "CLASS"]:
#     df = TargetEncoding(features=cate, encode_flag_column="encode_flag", target="target").fit_transform(df)
#
# # Convert toomany features: lump levels and map missings to own level
# if TARGET_TYPE in ["REGR", "CLASS"]:
#     df = MapToomany(features=toomany, n_top=10).fit_transform(df)

# Univariate variable importance
varimp_cate = calc_imp(df.query("fold != 'namethisutil'"), cate, target_type="REGR")
print(varimp_cate)

# Check
plot_distr(df.query("fold != 'namethisutil'"), cate, varimp=varimp_cate, target_type="REGR",
           color=color, ylim=ylim,
           nrow=2, ncol=3, w=18, h=12, pdf=plotloc + "distr_cate_h" + str(horizon) + ".pdf")


# --- Removing variables ---------------------------------------------------------------------------------------------

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df, cate, cutoff=cutoff_corr, n_cluster=5,
          pdf=plotloc + "corr_cate_h" + str(horizon) + ".pdf")


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Define final features ----------------------------------------------------------------------------------------
features = np.concatenate([metr, cate, toomany + "_ENCODED"])
features_binned = np.concatenate([setdiff(metr + "_BINNED_", onebin),
                                  setdiff(cate, "MISS_" + miss),
                                  toomany + "_ENCODED"])  # do not need indicators for binned
features_lgbm = np.append(metr, cate + "_ENCODED")

# Check
setdiff(features, df.columns.values.tolist())
setdiff(features_binned, df.columns.values.tolist())
setdiff(features_lgbm, df.columns.values.tolist())


# --- Remove burned data ----------------------------------------------------------------------------------------
df = df.query("fold != 'namethisutil'")


# --- Save image ------------------------------------------------------------------------------------------------------
plt.close(fig="all")  # plt.close(plt.gcf())
del df_orig

# Serialize
with open(TARGET_TYPE + "_1_explore.pkl", "wb") as file:
    pickle.dump({"df": df,
                 "metr": metr,
                 "cate": cate,
                 "features": features,
                 "features_binned": features_binned,
                 "features_lgbm": features_lgbm},
                file)
