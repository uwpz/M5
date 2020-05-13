
# ######################################################################################################################
#  Initialize: Libraries, functions, parameters
# ######################################################################################################################

# General libraries, parameters and functions
from initialize import *
# import sys; sys.path.append(getcwd() + "\\code") #not needed if code is marked as "source" in pycharm

# Specific libraries
from datetime import datetime

# Main parameter
horizon = 28

# Load results from exploration
with open("etl" + "_" + "n10000" + ".pkl", "rb") as file:
    d_pick = pickle.load(file)
df, df_tsfe, df_tsfe_sameweekday = d_pick["df"], d_pick["df_tsfe"], d_pick["df_tsfe_sameweekday"]


# ######################################################################################################################
#  Join depending on horizon
# ######################################################################################################################

# Filter on "filled data"
df = df.query("fold == 'train'").reset_index(drop = True)


# --- Join all and remove missing target------------------------------------------------------------------------

tmp = datetime.now()
df_h = (df.set_index(["date", "id"])
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


# --- Check metadata -------------------------------------------------------------------------------------------

df_meta = pd.read_excel("DATAMODEL_m5.xlsx")

# Check
print(setdiff(df_h.columns.values, df_meta["variable"].values))
print(setdiff(df_meta.loc[df_meta["status"] == "ready", "variable"].values, df_h.columns.values))
print(setdiff(df_h.columns.values, df_meta.loc[df_meta["status"] == "ready", "variable"].values))

# Filter on "ready"
df_meta_sub = df_meta.loc[df_meta["status"].isin(["ready"])]


########################################################################################################################
# Prepare final data
########################################################################################################################

# --- Save image ------------------------------------------------------------------------------------------------------
plt.close(fig="all")  # plt.close(plt.gcf())

# Serialize
with open("0_prepare_h" + str(horizon) + ".pkl", "wb") as file:
    pickle.dump({"df_h": df_h,
                 "df_meta_sub": df_meta_sub},
                file)
