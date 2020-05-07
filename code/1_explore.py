
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from scipy.stats.mstats import winsorize

# Main parameter
horizon = 9

# Specific parameters
cutoff_corr = 0.4
cutoff_varimp = 0.1
color = None

# Load results from exploration
with open("0_etl_h" + str(horizon) + ".pkl", "rb") as file:
    d_vars = pickle.load(file)
df, df_meta_sub = d_vars["df_train"], d_vars["df_meta_sub"]

# Check
# Check
print(setdiff(df.columns.values, df_meta_sub["variable"].values))
print(setdiff(df_meta_sub.loc[df_meta_sub["status"] == "ready", "variable"].values, df.columns.values))


# ######################################################################################################################
# ETL
# ######################################################################################################################

# Train/Test fold: usually split by time
np.random.seed(123)
df.loc[df["year"] == 2011, "myfold"] = "util"
df.groupby("myfold")["date"].describe()
df["myfold_num"] = df["myfold"].map({"train": 0, "util": 0, "test": 1})  # Used for pedicting test data
df["encode_flag"] = df["myfold"].map({"train": 0, "test": 0, "util": 1})  # Used for encoding


# ######################################################################################################################
# Metric variables: Explore and adapt
# ######################################################################################################################

# --- Define metric covariates -------------------------------------------------------------------------------------

metr = df_meta_sub.loc[df_meta_sub["type"] == "metr", "variable"].values
#df = Convert(features = metr, convert_to = "float").fit_transform(df)
df[metr].describe()

# --- Create nominal variables for all metric variables (for linear models) before imputing -------------------------
df[metr + "_BINNED"] = df[metr]
df = Binning(features = metr + "_BINNED").fit_transform(df)

# Convert missings to own level ("(Missing)")
df[metr + "_BINNED"] = df[metr + "_BINNED"].fillna("(Missing)")
print(create_values_df(df[metr + "_BINNED"], 11))

# Get binned variables with just 1 bin (removed later)
onebin = (metr + "_BINNED")[df[metr + "_BINNED"].nunique() == 1]
print(onebin)

# --- Missings + Outliers + Skewness ---------------------------------------------------------------------------------

# Remove covariates with too many missings from metr
misspct = df[metr].isnull().mean().round(3)  # missing percentage
print("misspct:\n", misspct.sort_values(ascending = False))  # view in descending order
remove = misspct[misspct > 0.95].index.values  # vars to remove
metr = setdiff(metr, remove)  # adapt metadata


# --- Final variable information ------------------------------------------------------------------------------------

# REGR
varimp_metr = calc_imp(df.sample(n = int(1e5)), np.append(metr, metr + "_BINNED"),
                       target = "demand", target_type = "REGR")
print(varimp_metr)
plot_distr(df.sample(n = int(1e5)), features = np.column_stack((metr, metr + "_BINNED")).ravel(),
           target = "demand", target_type = "REGR",
           varimp = varimp_metr, color = None, ylim = (0, 3), regplot = False,
           ncol = 4, nrow = 2, w = 24, h = 18,
           pdf = plotloc + "REGR_distr_metr_h" + str(horizon) + ".pdf")

# CLASS
varimp_metr = calc_imp(df.sample(n = int(1e5)), np.append(metr, metr + "_BINNED"),
                       target = "anydemand", target_type = "CLASS")
print(varimp_metr)
plot_distr(df.sample(n = int(1e5)), features = np.column_stack((metr, metr + "_BINNED")).ravel(),
           target = "anydemand", target_type = "CLASS",
           varimp = varimp_metr, color = twocol, ylim = None, regplot = False,
           ncol = 4, nrow = 2, w = 24, h = 18,
           pdf = plotloc + "CLASS_distr_metr_h" + str(horizon) + ".pdf")



# --- Removing variables -------------------------------------------------------------------------------------------

# Remove leakage features
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)

# Remove highly/perfectly (>=98%) correlated (the ones with less NA!)
df[metr].describe()
plot_corr(df.sample(n = int(1e5)), metr, cutoff = 0.9,
          w=12, h = 12, pdf = plotloc + "corr_metr.pdf")
remove = ["xxx", "xxx"]
metr = setdiff(metr, remove)


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!

# Univariate variable importance (again ONLY for non-missing observations!)
varimp_metr_fold = calc_imp(df.sample(n = int(1e5)), metr, target = "myfold_num")
print(varimp_metr_fold)

# --- Missing indicator and imputation (must be done at the end of all processing)------------------------------------

miss = metr[df[metr].isnull().any().values]  # alternative: [x for x in metr if df[x].isnull().any()]
#df["MISS_" + miss] = pd.DataFrame(np.where(df[miss].isnull(), "miss", "no_miss"))
#df["MISS_" + miss].describe()

# Impute missings with randomly sampled value (or median, see below)
df = DfSimpleImputer(features=miss, strategy="median").fit_transform(df)
df[miss].isnull().sum()


# ######################################################################################################################
# Categorical  variables: Explore and adapt
# ######################################################################################################################

# --- Define categorical covariates -----------------------------------------------------------------------------------

# Nominal variables
cate = df_meta_sub.loc[df_meta_sub["type"] == "cate", "variable"].values
df = Convert(features = cate, convert_to = "str").fit_transform(df)
df[cate].describe()

# Convert ordinal features to make it "alphanumerically sorted"
tmp = ["month"]
df[tmp] = df[tmp].apply(lambda x: x.str.zfill(2))


# --- Handling factor values ----------------------------------------------------------------------------------------

# Convert "standard" features: map missings to own level
df[cate] = df[cate].fillna("(Missing)")
df[cate].describe()

# Get "too many members" columns and copy these for additional encoded features (for tree based models)
topn_toomany = 40
levinfo = df[cate].nunique().sort_values(ascending = False)  # number of levels
print(levinfo)
toomany = levinfo[levinfo > topn_toomany].index.values
print(toomany)
toomany = setdiff(toomany, ["xxx", "xxx"])  # set exception for important variables

# Create encoded features (for tree based models), i.e. numeric representation
df = TargetEncoding(features = cate, encode_flag_column = "encode_flag", target = "demand").fit_transform(df)
#df["MISS_" + miss + "_ENCODED"] = df["MISS_" + miss].apply(lambda x: x.map({"no_miss": 0, "miss": 1}))

# Convert toomany features: lump levels and map missings to own level
df = MapToomany(features = toomany, n_top = 40).fit_transform(df)


# --- Final variable information ------------------------------------------------------------------------------------

# REGR
varimp_cate = calc_imp(df.sample(n = int(1e5)), cate, target = "demand", target_type = "REGR")
print(varimp_cate)
plot_distr(df.sample(n = int(1e5)), features = cate,
           target = "demand", target_type = "REGR",
           varimp = varimp_cate, color = None, ylim = (0, 3), regplot = False,
           ncol = 4, nrow = 2, w = 24, h = 18,
           pdf = plotloc + "REGR_distr_cate_h" + str(horizon) + ".pdf")

# CLASS
varimp_metr = calc_imp(df.sample(n = int(1e5)), cate, target = "anydemand", target_type = "CLASS")
print(varimp_cate)
plot_distr(df.sample(n = int(1e5)), features = cate,
           target = "anydemand", target_type = "CLASS",
           varimp = varimp_metr, color = twocol, ylim = None, regplot = False,
           ncol = 4, nrow = 2, w = 24, h = 18,
           pdf = plotloc + "CLASS_distr_cate_h" + str(horizon) + ".pdf")


# --- Removing variables ---------------------------------------------------------------------------------------------

# Remove highly/perfectly (>=99%) correlated (the ones with less levels!)
plot_corr(df.sample(n = int(1e5)), cate, cutoff = 0, n_cluster = 5,  # maybe plot miss separately
          w = 12, h = 12, pdf = plotloc + "corr_cate.pdf")


# --- Time/fold depedency --------------------------------------------------------------------------------------------

# Hint: In case of having a detailed date variable this can be used as regression target here as well!
# Univariate variable importance (again ONLY for non-missing observations!)
varimp_cate_fold = calc_imp(df.sample(n = int(1e5)), cate, target = "myfold_num")
print(varimp_cate_fold)


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Prepare ----------------------------------------------------------------------------------------
df = df.query("encode_flag == 0")
target_labels = "target"


# --- Define final features ----------------------------------------------------------------------------------------

# Standard: for xgboost or Lasso
metr_standard = np.append(metr, toomany + "_ENCODED")
cate_standard = cate

# Binned: for Lasso
metr_binned = np.array([])
cate_binned = np.append(setdiff(metr + "_BINNED", onebin), cate)

# Encoded: for Lightgbm or DeepLearning
metr_encoded = np.concatenate([metr, cate + "_ENCODED"])
cate_encoded = np.array([])

# Check
all_features = np.unique(np.concatenate([metr_standard, cate_standard, metr_binned, cate_binned, metr_encoded]))
setdiff(all_features, df.columns.values.tolist())
setdiff(df.columns.values.tolist(), all_features)


# --- Remove burned data ----------------------------------------------------------------------------------------

df = df.query("fold != 'util'").reset_index(drop = True)


# --- Save image ----------------------------------------------------------------------------------------------------

# Clean up
plt.close(fig = "all")  # plt.close(plt.gcf())

# Serialize
with open("1_explore_h" + str(horizon) + ".pkl", "wb") as file:
    pickle.dump({"df": df,
                 "target_labels": target_labels,
                 "metr_standard": metr_standard,
                 "cate_standard": cate_standard,
                 "metr_binned": metr_binned,
                 "cate_binned": cate_binned,
                 "metr_encoded": metr_encoded,
                 "cate_encoded": cate_encoded},
                file)