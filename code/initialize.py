
# ######################################################################################################################
# Libraries
# ######################################################################################################################

# Data
import numpy as np
import pandas as pd

# Plot
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns

# ETL
from scipy.stats import chi2_contingency
from scipy.sparse import hstack
from scipy.cluster.hierarchy import ward, dendrogram, fcluster

# ML
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


# Util
from collections import defaultdict
from os import getcwd
import pdb  # pdb.set_trace()  #quit with "q", next line with "n", continue with "c"
from joblib import Parallel, delayed
#from dill import (load_session, dump_session)
import pickle


# ######################################################################################################################
# Parameters
# ######################################################################################################################

# Locations
dataloc = "./data/"
plotloc = "./output/"

# Util
sns.set(style="whitegrid")
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 20)
# plt.ioff(); plt.ion()  # Interactive plotting? ion is default

# Other
twocol = ["red", "green"]
threecol = ["green", "yellow", "red"]
colors = pd.Series(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS))
colors = colors.iloc[np.setdiff1d(np.arange(len(colors)),[7,8,9,12,13,14,16,17,26])]
#sel = np.arange(50);  plt.bar(sel.astype("str"), 1, color=colors[sel])


# ######################################################################################################################
# My Functions and Classes
# ######################################################################################################################

# --- Functions ----------------------------------------------------------------------------------------

def setdiff(a, b):
    return np.setdiff1d(a, b, True)


def char_bins(series, n_bins=10):
    bins = pd.qcut(series, n_bins, duplicates="drop")
    bins.cat.categories = ["q" + str(i).zfill(2)  # + ":" + bins.cat.categories.astype("str")[i-1]
                           for i in 1 + np.arange(len(bins.cat.categories))]
    return bins.astype("str")


def spearman_loss_func(y_true, y_pred):
    spear = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).corr(method="spearman").values[0, 1]
    return spear


# Overview of values 
def create_values_df(df, topn):
    return pd.concat([df[catname].value_counts()[:topn].reset_index().
                     rename(columns={"index": catname, catname: catname + "_c"})
                      for catname in df.select_dtypes(["object"]).columns.values], axis=1)


# Show closed figure again
def show_figure(fig):
    # create a dummy figure and use its manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)
    

# Plot distribution regarding target
def plot_distr(df, features, target="target", target_type="CLASS", color=["blue", "red"], varimp=None, ylim=None,
               nrow=1, ncol=1, w=8, h=6, pdf=None):
    # df = df_test; features = cate; target = "residual"; target_type="REGR"; color=["blue", "red"]; varimp=None;
    # ylim = ylim_res; ncol=2; nrow=2; pdf=None; w=8; h=6

    # Help variables
    n_ppp = ncol * nrow  # plots per page
    n_pages = len(features) // n_ppp + 1  # number of pages
    pdf_pages = None

    # Open pdf
    if pdf is not None:
        pdf_pages = PdfPages(pdf)

    # Plot
    # l_fig = list()
    for page in range(n_pages):
        # page = 0
        fig, ax = plt.subplots(nrow, ncol)
        for i in range(n_ppp):
            # i = 1
            ax_act = ax.flat[i]
            if page * n_ppp + i <= max(range(len(features))):
                feature_act = features[page * n_ppp + i]

                # Categorical feature
                if df[feature_act].dtype == "object":
                    # Prepare data
                   # pdb.set_trace()
                    df_plot = pd.DataFrame({"h": df.groupby(feature_act)[target].mean(),
                                            "w": df.groupby(feature_act).size()}).reset_index()
                    df_plot["pct"] = 100 * df_plot["w"] / len(df)
                    df_plot["w"] = 0.9 * df_plot["w"] / max(df_plot["w"])
                    df_plot[feature_act + "_new"] = df_plot[feature_act] + " (" + \
                                                    (df_plot["pct"]).round(1).astype(str) + "%)"
                    df_plot["new_w"] = np.where(df_plot["w"].values < 0.2, 0.2, df_plot["w"])
                    if target_type == "CLASS":
                        # Target barplot
                        # sns.barplot(df_tmp.h, df_tmp[cate[page * ppp + i]], orient="h", color="coral", ax=axact)
                        ax_act.barh(df_plot[feature_act], df_plot.h, height=df_plot.new_w,
                                    color=color[1], edgecolor="black", alpha=0.5, linewidth=1)
                        ax_act.set_xlabel("mean(" + target + ")")
                    if target_type == "REGR":
                        if ylim is not None:
                            ax_act.set_xlim(ylim)

                        # Target boxplot
                        bp = df[[feature_act, target]].boxplot(target, feature_act, vert=False,
                                                                        widths=df_plot.w.values,
                                                                        showmeans=True,
                                                                        meanprops=dict(marker="x",
                                                                                       markeredgecolor="black"),
                                                                        flierprops=dict(marker="."),
                                                                        return_type='dict',
                                                                        ax=ax_act)
                        [[item.set_color('black') for item in bp[target][key]] for key in bp[target].keys()]
                        fig.suptitle("")
                        ax_act.set_xlabel(target)
                    ax_act.set_yticklabels(df_plot[feature_act + "_new"].values)
                    if varimp is not None:
                        ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                    ax_act.axvline(np.mean(df[target]), ls="dotted", color="black")

                    # Inner barplot
                    xlim = ax_act.get_xlim()
                    ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
                    inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
                    inset_ax.set_axis_off()
                    ax_act.axvline(0, color="black")
                    if target_type == "CLASS":
                        #inset_ax.set_yticklabels(df_plot[feature_act + "_new"].values)
                        #ax_act.get_shared_y_axes().join(ax_act, inset_ax)
                        inset_ax.barh(df_plot[feature_act], df_plot.w,
                                      color="lightgrey", edgecolor="black", linewidth=1)
                    if target_type == "REGR":
                        df_plot.plot.barh(y="w", x=feature_act,
                                          color="lightgrey", ax=inset_ax, edgecolor="black", linewidth=1, legend=False)
                        # inset_ax.barh(df_plot[feature_act], df_plot.w, color="lightgrey", edgecolor="black",
                        #              linewidth=1)

                # Metric feature
                else:
                    if target_type == "CLASS":
                        sns.distplot(df.loc[df[target] == 0, feature_act].dropna(), color=color[0],
                                     bins=20, label="0", ax=ax_act)
                        sns.distplot(df.loc[df[target] == 1, feature_act].dropna(), color=color[1],
                                     bins=20, label="1", ax=ax_act)
                        # sns.FacetGrid(df, hue=target, palette=["red","blue"])\
                        #     .map(sns.distplot, metr[i])\
                        #     .add_legend() # does not work for multiple axes
                        if varimp is not None:
                            ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                        else:
                            ax_act.set_title(feature_act)
                        ax_act.set_ylabel("density")
                        ax_act.set_xlabel(feature_act + "(NA: " +
                                          str((df[feature_act].isnull().mean() * 100).round(1)) +
                                          "%)")

                        # Inner Boxplot
                        ylim = ax_act.get_ylim()
                        ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
                        inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
                        inset_ax.set_axis_off()
                        ax_act.get_shared_x_axes().join(ax_act, inset_ax)
                        i_bool = df[feature_act].notnull()
                        sns.boxplot(x=df.loc[i_bool, feature_act],
                                    y=df.loc[i_bool, target].astype("category"),
                                    showmeans=True,
                                    meanprops={"marker": "x", "markerfacecolor": "black", "markeredgecolor":"black"},
                                    palette=color,
                                    ax=inset_ax)
                        ax_act.legend(title=target, loc="best")

                    if target_type == "REGR":
                        if ylim is not None:
                            ax_act.set_ylim(ylim)
                            tmp_scale = (ylim[1] - ylim[0]) / (np.max(df[target]) - np.min(df[target]))
                        else:
                            tmp_scale = 1
                        tmp_cmap = mcolors.LinearSegmentedColormap.from_list("wh_bl_yl_rd",
                                                                            [(1, 1, 1, 0), "blue", "yellow", "red"])
                        ax_act.set_facecolor('0.98')
                        p = ax_act.hexbin(df[feature_act], df[target],
                                          gridsize=(int(50 * tmp_scale), 50),
                                          cmap=tmp_cmap)
                        plt.colorbar(p, ax=ax_act)
                        sns.regplot(feature_act, target, df, lowess=True, scatter=False, color="black", ax=ax_act)
                        if varimp is not None:
                            ax_act.set_title(feature_act + " (VI:" + str(varimp[feature_act]) + ")")
                        else:
                            ax_act.set_title(feature_act)
                        ax_act.set_ylabel(target)
                        ax_act.set_xlabel(feature_act + "(NA: " +
                                          str(df[feature_act].isnull().mean().round(3) * 100) +
                                          "%)")
                        ylim = ax_act.get_ylim()
                        # ax_act.grid(False)
                        ax_act.axhline(color="grey")

                        # Inner Histogram
                        ax_act.set_ylim(ylim[0] - 0.4 * (ylim[1] - ylim[0]))
                        inset_ax = ax_act.inset_axes([0, 0.05, 1, 0.15])
                        inset_ax.set_axis_off()
                        inset_ax.get_yaxis().set_visible(False)
                        ax_act.get_shared_x_axes().join(ax_act, inset_ax)
                        i_bool = df[feature_act].notnull()
                        sns.distplot(df[feature_act].dropna(), bins=20, color="black", ax=inset_ax)
                        inset_ax.set_xlabel("")

                        # Inner-inner Boxplot
                        #ylim_inner = inset_ax.get_ylim()
                        #inset_ax.set_ylim(ylim_inner[0] - 0.35 * (ylim_inner[1] - ylim_inner[0]))
                        inset_ax = ax_act.inset_axes([0, 0, 1, 0.05])
                        #inset_ax.set_axis_off()
                        inset_ax.get_shared_x_axes().join(ax_act, inset_ax)
                        sns.boxplot(x=df.loc[i_bool, feature_act], palette=["grey"], ax=inset_ax)
                        #inset_ax.set_xlabel("")
                        inset_ax.set_xlabel(feature_act + "(NA: " +
                                          str(df[feature_act].isnull().mean().round(3) * 100) +
                                          "%)")
                        # plt.show()
            else:
                ax_act.axis("off")  # Remove left-over plots
        # plt.subplots_adjust(wspace=1)
        fig.set_size_inches(w=w, h=h)
        fig.tight_layout()
        # l_fig.append(fig)
        if pdf is not None:
            pdf_pages.savefig(fig)
            # plt.close(fig)
        plt.show()
    if pdf is not None:
        pdf_pages.close()


# Plot correlation
def plot_corr(df, features, cutoff=0, n_cluster=5, w=8, h=6, pdf=None):
    # df = df; features = cate; cutoff = 0.1; w=8; h=6; pdf="blub.pdf"

    metr = features[df[features].dtypes != "object"]
    cate = features[df[features].dtypes == "object"]
    df_corr = None

    if len(metr) and len(cate):
        raise Exception('Mixed dtypes')
        # return

    if len(cate):
        df_corr = pd.DataFrame(np.ones([len(cate), len(cate)]), index=cate, columns=cate)
        for i in range(len(cate)):
            print("cate=", cate[i])
            for j in range(i+1, len(cate)):
                # i=1; j=2
                tmp = pd.crosstab(df[features[i]], df[features[j]])
                n = np.sum(tmp.values)
                m = min(tmp.shape)
                chi2 = chi2_contingency(tmp)[0]
                df_corr.iloc[i, j] = np.sqrt(chi2 / (n + chi2)) * np.sqrt(m / (m-1))
                df_corr.iloc[j, i] = df_corr.iloc[i, j]
        d_new_names = dict(zip(df_corr.columns.values,
                               df_corr.columns.values + " (" +
                               df[df_corr.columns.values].nunique().astype("str").values + ")"))
        df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

    if len(metr):
        df_corr = abs(df[metr].corr(method="spearman"))
        d_new_names = dict(zip(df_corr.columns.values,
                               df_corr.columns.values + " (NA: " +
                               (df[df_corr.columns.values].isnull().mean() * 100).round(1).astype("str").values + "%)"))
        df_corr.rename(columns=d_new_names, index=d_new_names, inplace=True)

    # Cut off
    np.fill_diagonal(df_corr.values, 0)
    i_bool = (df_corr.max(axis=1) > cutoff).values
    df_corr = df_corr.loc[i_bool, i_bool]
    np.fill_diagonal(df_corr.values, 1)

    # Cluster df_corr
    new_order = df_corr.columns.values[fcluster(ward(1 - np.triu(df_corr)), n_cluster, criterion='maxclust').argsort()]
    df_corr = df_corr.loc[new_order, new_order]

    # Plot
    fig, ax = plt.subplots(1, 1)
    ax_act = ax
    sns.heatmap(df_corr, annot=True, fmt=".2f", cmap="Blues", ax=ax_act)
    ax_act.set_yticklabels(labels=ax_act.get_yticklabels(), rotation=0)
    ax_act.set_xticklabels(labels=ax_act.get_xticklabels(), rotation=90)
    if len(metr):
        ax_act.set_title("Absolute spearman correlation (cutoff at " + str(cutoff) +")")
    if len(cate):
        ax_act.set_title("Contingency coefficient (cutoff at " + str(cutoff) + ")")
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        # plt.close(fig)
    plt.show()


# Univariate variable importance
def calc_imp(df, features, target="target", target_type="CLASS"):
    # df=df; features=metr; target="target"; target_type="CLASS"
    #pdb.set_trace()
    varimp = pd.Series()
    for feature_act in features:
        # feature_act=metr[8]
        if target_type == "CLASS":
            if df[feature_act].dtype == "object":
                varimp_act = {feature_act: (roc_auc_score(y_true=df[target].values,
                                                          y_score=df[[feature_act, target]]
                                                          .groupby(feature_act)[target]
                                                          .transform("mean").values)
                                            .round(3))}
            else:
                try:
                    varimp_act = {feature_act: (roc_auc_score(y_true=df[target].values,
                                                              y_score=df[[target]]
                                                              .assign(dummy=pd.qcut(df[feature_act], 10,
                                                                                    duplicates="drop")
                                                                      .astype("object")
                                                                      .fillna("(Missing)"))
                                                              .groupby("dummy")[target]
                                                              .transform("mean").values)
                                                .round(3))}
                except:
                    varimp_act = {feature_act: 0.5}

        if target_type == "REGR":
            if df[feature_act].dtype == "object":
                df_tmp = df[[feature_act, target]]\
                    .assign(grouped_mean=lambda x: x.groupby(feature_act)[target].transform("mean"))
                varimp_act = {feature_act: (abs(df_tmp[["grouped_mean", target]].corr(method="pearson").values[0, 1])
                                            .round(3))}
            else:
                varimp_act = {feature_act: (abs(df[[feature_act, target]].corr(method="pearson").values[0, 1])
                                            .round(3))}

        varimp = varimp.append(pd.Series(varimp_act))
    varimp.sort_values(ascending=False, inplace=True)
    return varimp


# # Undersample
# def undersample_n(df, n_max_per_level, target="target", random_state=42):
#     b_all = df[target].mean()
#     df = df.groupby(target).apply(lambda x: x.sample(min(n_max_per_level, x.shape[0]),
#                                                      random_state=random_state)) \
#         .reset_index(drop=True)
#     b_sample = df[target].mean()
#     return df, b_sample, b_all


# Rescale predictions (e.g. to rewind undersampling)
def scale_predictions(yhat, b_sample=None, b_all=None):
    if b_sample is None:
        yhat_rescaled = yhat
    else:
        tmp = yhat * np.array([1 - b_all, b_all]) / np.array([1 - b_sample, b_sample])
        yhat_rescaled = (tmp.T / tmp.sum(axis=1)).T
    return yhat_rescaled


# Plot ML-algorithm performance
def plot_all_performances(y, yhat, labels=None, target_type="CLASS", colors=None, ylim=None, w=18, h=12, pdf=None):
    # y=df_test["target"]; yhat=yhat_test; ylim = None; w=12; h=8

    if target_type == "CLASS":
        fig, ax = plt.subplots(2, 3)

        # Roc curve
        ax_act = ax[0, 0]
        fpr, tpr, cutoff = roc_curve(y, yhat[:, 1])
        roc_auc = roc_auc_score(y, yhat[:, 1])
        sns.lineplot(fpr, tpr, ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': r"fpr: P($\^y$=1|$y$=0)",
                 'ylabel': r"tpr: P($\^y$=1|$y$=1)",
                 'title': "ROC (AUC = {0:.2f})".format(roc_auc)}
        ax_act.set(**props)

        # Confusion matrix
        ax_act = ax[0, 1]
        df_conf = pd.DataFrame(confusion_matrix(y, np.where(yhat[:, 1] > 0.5, 1, 0)))
        acc = accuracy_score(y, np.where(yhat[:, 1] > 0.5, 1, 0))
        sns.heatmap(df_conf, annot=True, fmt=".5g", cmap="Greys", ax=ax_act)
        props = {'xlabel': "Predicted label",
                 'ylabel': "True label",
                 'title': "Confusion Matrix (Acc ={0: .2f})".format(acc)}
        ax_act.set(**props)

        # Distribution plot
        ax_act = ax[0, 2]
        sns.distplot(yhat[:, 1][y == 1], color="red", label="1", bins=20, ax=ax_act)
        sns.distplot(yhat[:, 1][y == 0], color="blue", label="0", bins=20, ax=ax_act)
        props = {'xlabel': r"Predictions ($\^y$)",
                 'ylabel': "Density",
                 'title': "Distribution of Predictions",
                 'xlim': (0, 1)}
        ax_act.set(**props)
        ax_act.legend(title="Target", loc="best")

        # Calibration
        ax_act = ax[1, 0]
        true, predicted = calibration_curve(y, yhat[:, 1], n_bins=10)
        sns.lineplot(predicted, true, ax=ax_act, marker="o")
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)

        # Precision Recall
        ax_act = ax[1, 1]
        prec, rec, cutoff = precision_recall_curve(y, yhat[:, 1])
        prec_rec_auc = average_precision_score(y, yhat[:, 1])
        sns.lineplot(rec, prec, ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': r"recall=tpr: P($\^y$=1|$y$=1)",
                 'ylabel': r"precision: P($y$=1|$\^y$=1)",
                 'title': "Precision Recall Curve (AUC = {0:.2f})".format(prec_rec_auc)}
        ax_act.set(**props)
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate("{0: .1f}".format(thres), (rec[i_thres], prec[i_thres]), fontsize=10)

        # Precision
        ax_act = ax[1, 2]
        pct_tested = np.array([])
        for thres in cutoff:
            pct_tested = np.append(pct_tested, [np.sum(yhat[:, 1] >= thres)/len(yhat)])
        sns.lineplot(pct_tested, prec[:-1], ax=ax_act, palette=sns.xkcd_palette(["red"]))
        props = {'xlabel': "% Samples Tested",
                 'ylabel': r"precision: P($y$=1|$\^y$=1)",
                 'title': "Precision Curve"}
        ax_act.set(**props)
        for thres in np.arange(0.1, 1, 0.1):
            i_thres = np.argmax(cutoff > thres)
            ax_act.annotate("{0: .1f}".format(thres), (pct_tested[i_thres], prec[i_thres]), fontsize=10)

    if target_type == "REGR":
        fig, ax = plt.subplots(2, 3)

        def plot_scatter(x, y, xlabel="x", ylabel="y", title=None, ylim=None, ax_act=None):
            if ylim is not None:
                ax_act.set_ylim(ylim)
                tmp_scale = (ylim[1] - ylim[0]) / (np.max(y) - np.min(y))
            else:
                tmp_scale = 1
            tmp_cmap = mcolors.LinearSegmentedColormap.from_list("wh_bl_yl_rd",
                                                                [(1, 1, 1, 0), "blue", "yellow", "red"])
            p = ax_act.hexbin(x, y,
                              gridsize=(int(50 * tmp_scale), 50),
                              cmap=tmp_cmap)
            plt.colorbar(p, ax=ax_act)
            sns.regplot(x, y, lowess=True, scatter=False, color="black", ax=ax_act)
            ax_act.set_title(title)
            ax_act.set_ylabel(ylabel)
            ax_act.set_xlabel(xlabel)

            ax_act.set_facecolor('white')
            # ax_act.grid(False)

            ylim = ax_act.get_ylim()
            xlim = ax_act.get_xlim()

            # Inner Histogram on y
            ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))
            inset_ax = ax_act.inset_axes([0, 0, 0.2, 1])
            inset_ax.set_axis_off()
            ax_act.get_shared_y_axes().join(ax_act, inset_ax)
            sns.distplot(y, color="grey", vertical=True, ax=inset_ax)

            # Inner-inner Boxplot on y
            xlim_inner = inset_ax.get_xlim()
            inset_ax.set_xlim(xlim_inner[0] - 0.3 * (xlim_inner[1] - xlim_inner[0]))
            inset_inset_ax = inset_ax.inset_axes([0, 0, 0.2, 1])
            inset_inset_ax.set_axis_off()
            inset_ax.get_shared_y_axes().join(inset_ax, inset_inset_ax)
            sns.boxplot(y, palette=["grey"], orient="v", ax=inset_inset_ax)

            # Inner Histogram on x
            ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
            inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
            inset_ax.set_axis_off()
            ax_act.get_shared_x_axes().join(ax_act, inset_ax)
            sns.distplot(x, color="grey", ax=inset_ax)

            # Inner-inner Boxplot on x
            ylim_inner = inset_ax.get_ylim()
            inset_ax.set_ylim(ylim_inner[0] - 0.3 * (ylim_inner[1] - ylim_inner[0]))
            inset_inset_ax = inset_ax.inset_axes([0, 0, 1, 0.2])
            inset_inset_ax.set_axis_off()
            inset_ax.get_shared_x_axes().join(inset_ax, inset_inset_ax)
            sns.boxplot(x, palette=["grey"], ax=inset_inset_ax)

            ax_act.set_xlim(xlim[0] - 0.3 * (xlim[1] - xlim[0]))  # need to set again

        # Scatter plots
        plot_scatter(yhat, y,
                     xlabel=r"$\^y$", ylabel="y",
                     title=r"Observed vs. Fitted ($\rho_{Spearman}$ = " +
                           str(spearman_loss_func(y, yhat).round(3)) + ")",
                     ylim=ylim, ax_act=ax[0, 0])
        plot_scatter(yhat, y - yhat,
                     xlabel=r"$\^y$", ylabel=r"y-$\^y$", title="Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 0])
        plot_scatter(yhat, abs(y - yhat),
                     xlabel=r"$\^y$", ylabel=r"|y-$\^y$|", title="Absolute Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 1])
        plot_scatter(yhat, abs(y - yhat) / abs(y),
                     xlabel=r"$\^y$", ylabel=r"|y-$\^y$|/|y|", title="Relative Residuals vs. Fitted",
                     ylim=ylim, ax_act=ax[1, 2])

        # Calibration
        ax_act = ax[0, 1]
        df_calib = pd.DataFrame({"y": y, "yhat": yhat})\
            .assign(bin=lambda x: pd.qcut(x["yhat"], 10, duplicates="drop").astype("str"))\
            .groupby(["bin"], as_index=False).agg("mean")\
            .sort_values("yhat")
        sns.lineplot("yhat", "y", data=df_calib, ax=ax_act, marker="o")
        props = {'xlabel': r"$\bar{\^y}$ in $\^y$-bin",
                 'ylabel': r"$\bar{y}$ in $\^y$-bin",
                 'title': "Calibration"}
        ax_act.set(**props)

        # Distribution
        ax_act = ax[0, 2]
        sns.distplot(y, color="blue", label="y", ax=ax_act)
        sns.distplot(yhat, color="red", label=r"$\^y$", ax=ax_act)
        ax_act.set_ylabel("density")
        ax_act.set_xlabel("")
        ax_act.set_title("Distribution")

        ylim = ax_act.get_ylim()
        ax_act.set_ylim(ylim[0] - 0.3 * (ylim[1] - ylim[0]))
        inset_ax = ax_act.inset_axes([0, 0, 1, 0.2])
        inset_ax.set_axis_off()
        ax_act.get_shared_x_axes().join(ax_act, inset_ax)
        df_distr = pd.concat([pd.DataFrame({"type": "y", "values": y}),
                              pd.DataFrame({"type": "yhat", "values": yhat})])
        sns.boxplot(x=df_distr["values"],
                    y=df_distr["type"].astype("category"),
                    # order=df[feature_act].value_counts().index.values[::-1],
                    palette=["blue", "red"],
                    ax=inset_ax)
        ax_act.legend(title="", loc="best")

    if target_type == "MULTICLASS":
        fig, ax = plt.subplots(2, 2)

        # AUC
        k = yhat.shape[1]
        ax_act = ax[0, 0]
        aucs = np.array([roc_auc_score(np.where(y == i, 1, 0), yhat[:, i]).round(2) for i in np.arange(k)])

        for i in np.arange(k):
            #  i=3
            y_bin = np.where(y == i, 1, 0)
            fpr, tpr, cutoff = roc_curve(y_bin, yhat[:, i])
            new_label = labels[i] + " (" + str(aucs[i]) + ")"
            ax_act.plot(fpr, tpr, color=colors[i], label=new_label)
        mean_auc = np.average(aucs).round(3)
        weighted_auc = np.average(aucs, weights=np.array(np.unique(y, return_counts=True))[1, :]).round(3)
        props = dict(xlabel=r"fpr: P($\^y$=1|$y$=0)",
                     ylabel=r"tpr: P($\^y$=1|$y$=1)",
                     title="ROC\n" + r"($AUC_{mean}$ = " + str(mean_auc) + r", $AUC_{weighted}$ = " +
                           str(weighted_auc) + ")")
        ax_act.set(**props)
        ax_act.legend(title=r"Target ($AUC_{1-vs-all}$)", loc='best')

        # Confusion Matrix
        ax_act = ax[1, 0]

        y_pred = yhat.argmax(axis=1)
        freq_true = np.unique(y_true, return_counts=True)[1]
        freqpct_true = np.round(np.divide(freq_true, len(y_true)) * 100, 1)
        freq_pred = np.unique(y_pred, return_counts=True)[1]
        freqpct_pred = np.round(np.divide(freq_pred, len(y_true)) * 100, 1)

        m_conf = confusion_matrix(y, y_pred)
        prec = np.round(np.diag(m_conf) / m_conf.sum(axis=1) * 100, 1)
        rec = np.round(np.diag(m_conf) / m_conf.sum(axis=0) * 100, 1)
        f1 = np.round(2 * prec * rec / (prec + rec), 1)

        ylabels = [labels[i] + " (" + str(freq_true[i]) + ": " + str(freqpct_true[i]) + "%)" for i in
                   np.arange(len(labels))]
        xlabels = [labels[i] + " (" + str(freq_pred[i]) + ": " + str(freqpct_pred[i]) + "%)" for i in
                   np.arange(len(labels))]
        df_conf = pd.DataFrame(m_conf, columns=labels, index=labels).rename_axis(index="True label",
                                                                                 columns="Predicted label")
        acc = accuracy_score(y, y_pred)
        sns.heatmap(df_conf, annot=True, fmt=".5g", cmap="Blues", ax=ax_act, xticklabels=True, yticklabels=True)
        ax_act.set_yticklabels(labels=ylabels, rotation=0)
        ax_act.set_xticklabels(labels=xlabels, rotation=90, ha="center")
        props = dict(ylabel="True label (#: %)",
                     xlabel="Predicted label (#: %)",
                     title="Confusion Matrix (Acc ={0: .2f})".format(acc))
        ax_act.set(**props)
        for text in ax_act.texts[::len(labels) + 1]:
            text.set_weight('bold')

        # Metrics
        ax_act = ax[0, 1]
        df_metrics = pd.DataFrame(np.column_stack((y, np.flip(np.argsort(yhat, axis=1), axis=1)[:, :3])),
                                  columns=["y", "yhat1", "yhat2", "yhat3"]). \
            assign(acc_top1=lambda x: (x["y"] == x["yhat1"]).astype("int"),
                   acc_top2=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"])).astype("int"),
                   acc_top3=lambda x: ((x["y"] == x["yhat1"]) | (x["y"] == x["yhat2"]) |
                                       (x["y"] == x["yhat3"])).astype("int")). \
            assign(label=lambda x: np.array(labels, dtype="object")[x["y"].values]). \
            groupby(["label"])["acc_top1", "acc_top2", "acc_top3"].agg("mean").round(2). \
            join(pd.DataFrame(np.stack((aucs, rec, prec, f1), axis=1),
                              index=labels, columns=["auc", "recall", "precision", "f1"]))

        sns.heatmap(df_metrics, annot=True, fmt=".5g",
                    cmap=ListedColormap(['white']), linewidths=2, linecolor="black", cbar=False,
                    ax=ax_act, xticklabels=True, yticklabels=True)
        ax_act.set_xticklabels(labels=['Accuracy\n Top1', 'Accuracy\n Top2', 'Accuracy\n Top3', "AUC\n 1-vs-all",
                                       'Recall\n' r"P($\^y$=k|$y$=k))", 'Precision\n' r"P($y$=k|$\^y$=k))", 'F1'])
        ax_act.xaxis.tick_top()  # x axis on top
        ax_act.xaxis.set_label_position('top')
        ax_act.tick_params(left=False, top=False)
        ax_act.set_ylabel("True label")

        # Barplot
        ax_act = ax[1, 1]
        df_conf.iloc[::-1].plot.barh(stacked=True, ax=ax_act, color=colors[:len(labels)])
        ax_act.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax_act.legend(loc='center left', title="Predicted label", bbox_to_anchor=(1, 0.5))

    # Adapt figure
    fig.set_size_inches(w=w, h=h)
    fig.tight_layout()
    if pdf is not None:
        fig.savefig(pdf)
        # plt.close(fig)
    plt.show()


# Variable importance
def calc_varimp_by_permutation(df, df_ref, fit,
                               target, metr, cate,
                               target_type="CLASS",
                               b_sample=None, b_all=None,
                               features=None,
                               n_jobs=8):
    # df=df_train;  df_ref=df; target = "target"
    all_features = np.append(metr, cate)
    if features is None:
        features = all_features

    # Original performance
    if target_type == "CLASS":
        perf_orig = roc_auc_score(df[target],
                                  scale_predictions(fit.predict_proba(CreateSparseMatrix(metr, cate, df_ref).
                                                                      fit_transform(df)),
                                                    b_sample, b_all)[:, 1])
    if target_type == "REGR":
        perf_orig = spearman_loss_func(df[target],
                                       fit.predict(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df)))

    # Performance per variable after permutation
    i_perm = np.random.permutation(np.arange(len(df)))  # permutation vector

    # TODO: add all variables used as parameter
    def run_in_parallel(df_perm, feature):
        df_perm[feature] = df_perm[feature].values[i_perm]
        if target_type == "CLASS":
            perf = roc_auc_score(df_perm[target],
                                 scale_predictions(
                                     fit.predict_proba(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df_perm)),
                                     b_sample, b_all)[:, 1])
        if target_type == "REGR":
            perf = spearman_loss_func(df_perm[target],
                                      fit.predict(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df_perm)))
        return perf
    perf = Parallel(n_jobs=n_jobs)(delayed(run_in_parallel)(df, feature)
                                   for feature in features)

    # Collect performances and calculate importance
    df_varimp = pd.DataFrame({"feature": features, "perf_diff": np.maximum(0, perf_orig - perf)})\
        .sort_values(["perf_diff"], ascending=False)\
        .assign(importance=lambda x: 100 * x["perf_diff"] / max(x["perf_diff"]))\
        .assign(importance_cum=lambda x: 100 * x["perf_diff"].cumsum() / sum(x["perf_diff"]))\
        .assign(importance_sumnormed=lambda x: 100 * x["perf_diff"] / sum(x["perf_diff"]))

    return df_varimp


# Partial dependence
def calc_partial_dependence(df, df_ref, fit,
                            target, metr, cate,
                            target_type="CLASS",
                            b_sample=None, b_all=None,
                            features=None,
                            quantiles=np.arange(0, 1.1, 0.1),
                            n_jobs=8):
    # df=df_test;  df_ref=df_traintest; target = "target"; target_type=TARGET_TYPE; features=None;
    # quantiles = np.arange(0, 1.1, 0.1);n_jobs=4
    all_features = np.append(metr, cate)
    if features is None:
        features = all_features

    df_pd = pd.DataFrame()

    def run_in_parallel(feature):
        # feature = features[3]
        if feature in metr:
            values = df_ref[feature].quantile(quantiles)
        else:
            values = np.sort(df_ref[feature].unique())

        df_tmp = df.copy()  # save original data

        df_pd_feature = pd.DataFrame()
        for value in values:
            #value=values[0]
            df_tmp[feature] = value
            if target_type == "CLASS":
                yhat_mean = np.mean(scale_predictions(fit.predict_proba(CreateSparseMatrix(metr, cate, df_ref).
                                                                        fit_transform(df_tmp)),
                                    b_sample, b_all)[:, 1])
            if target_type == "REGR":
                yhat_mean = np.mean(fit.predict(CreateSparseMatrix(metr, cate, df_ref).fit_transform(df_tmp)),
                                    b_sample, b_all)

            df_pd_feature = pd.concat([df_pd_feature,
                                       pd.DataFrame({"feature": feature, "value": str(value), "yhat_mean": yhat_mean},
                                                    index=[0])])
        return df_pd_feature
    df_pd = pd.concat(Parallel(n_jobs=n_jobs)(delayed(run_in_parallel)(feature)
                                              for feature in features))
    return df_pd


# --- Classes ----------------------------------------------------------------------------------------

'''
# Special Kfold splitter: training fold only from training data, test fold only from test data
class KFoldTraintestSep:
    def __init__(self, n_splits=3, random_state=None):
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X, y, folds=None):
        for i_train, i_test in KFold(n_splits=self.n_splits,
                                     shuffle=True,
                                     random_state=self.random_state).split(X, y):
            i_train_train = i_train[folds[i_train] == "train"]
            i_test_test = i_test[folds[i_test] == "test"]
            yield i_train_train, i_test_test

    def get_n_splits(self):
        return self.n_splits
'''

# Special splitter: training fold only from training data, test fold only from test data
class TrainTestSep:
    def __init__(self, n_splits=1, sample_type="cv", fold_var="fold", random_state=42):
        self.n_splits = n_splits
        self.sample_type = sample_type
        self.fold_var = fold_var
        self.random_state = random_state

    def split(self, df):
        i_df = np.arange(len(df))
        np.random.seed(self.random_state)
        np.random.shuffle(i_df)
        i_train = i_df[df[self.fold_var].values[i_df] == "train"]
        i_test = i_df[df[self.fold_var].values[i_df] == "test"]
        if self.sample_type == "cv":
            splits_train = np.array_split(i_train, self.n_splits)
            splits_test = np.array_split(i_test, self.n_splits)
        else:
            splits_train = None
            splits_test = None
        for i in range(self.n_splits):
            if self.sample_type == "cv":
                i_train_yield = np.concatenate(splits_train)
                if self.n_splits > 1:
                    i_train_yield = np.setdiff1d(i_train_yield, splits_train[i], assume_unique=True)
                i_test_yield = splits_test[i]
            elif self.sample_type == "bootstrap":
                np.random.seed(self.random_state * (i + 1))
                i_train_yield = np.random.choice(i_train, len(i_train))
                np.random.seed(self.random_state * (i + 1))
                i_test_yield = np.random.choice(i_test, len(i_test))
            else:
                i_train_yield = None
                i_test_yield = None
            yield i_train_yield, i_test_yield

    def get_n_splits(self):
        return self.n_splits


# Map Nonexisting members of a string column to modus
class MapNonexisting(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features
        self._d_unique = None
        self._d_modus = None

    def fit(self, df):
        self._d_unique = {x: pd.unique(df[x]) for x in self.features}
        self._d_modus = {x: df[x].value_counts().index[0] for x in self.features}
        return self

    def transform(self, df):
        df = df.apply(lambda x: x.where(np.in1d(x, self._d_unique[x.name]),
                                        self._d_modus[x.name]) if x.name in self.features else x)
        return df

    def fit_transform(self, df, y=None, **fit_params):
        if fit_params["transform"]:
            return self.fit(df).transform(df)
        else:
            self.fit(df)
            return df


# Map Non-topn frequent members of a string column to "other" label
class MapToomany(BaseEstimator, TransformerMixin):
    def __init__(self, features, n_top=10, other_label="_OTHER_"):
        self.features = features
        self.other_label = other_label
        self.n_top = n_top
        self._s_levinfo = None
        self._toomany = None
        self._d_top = None
        self._statistics = None

    def fit(self, df, y=None):
        self._s_levinfo = df[self.features].apply(lambda x: x.unique().size).sort_values(ascending=False)
        self._toomany = self._s_levinfo[self._s_levinfo > self.n_top].index.values
        self._d_top = {x: df[x].value_counts().index.values[:self.n_top] for x in self._toomany}
        self._statistics = {"_s_levinfo": self._s_levinfo, "_toomany": self._toomany, "_d_top": self._d_top}
        return self

    def transform(self, df):
        df = df.apply(lambda x: x.where(np.in1d(x, self._d_top[x.name]),
                                        other=self.other_label) if x.name in self._toomany else x)
        return df


# Target Encoding
class TargetEncoding(BaseEstimator, TransformerMixin):
    def __init__(self, features, encode_flag_column="use_for_encoding", target="target"):
        self.features = features
        self.encode_flag_column = encode_flag_column
        self.target = target
        self._d_map = None
        self._statistics = None

    def fit(self, df, y=None):
        self._d_map = {x: df.loc[df[self.encode_flag_column] == 1, :]
                            .groupby(x, as_index=False)[self.target].agg("mean")
                            .sort_values(self.target, ascending=False)
                            .assign(rank=lambda x: np.arange(len(x)) + 1)
                            .set_index(x)
                            ["rank"]
                            .to_dict() for x in self.features}
        self._statistics = {"_d_map": self._d_map}
        return self

    def transform(self, df):
        #pdb.set_trace()
        df[self.features + "_ENCODED"] = df[self.features].apply(lambda x: x.map(self._d_map[x.name])
                                                                 .fillna(np.median(list(self._d_map[x.name].values()))))
        if self.encode_flag_column in df.columns.values:
            return df.loc[df[self.encode_flag_column] != 1, :]
        else:
            return df


# SimpleImputer for data frames
class DfSimpleImputer(SimpleImputer):
    def __init__(self, features, **kwargs):
        super().__init__(**kwargs)
        self.features = features

    def fit(self, df, y=None, **kwargs):
        fit = super().fit(df[self.features], **kwargs)
        return fit

    def transform(self, df):
        df[self.features] = super().transform(df[self.features].values)
        return df


# Convert
class Convert(BaseEstimator, TransformerMixin):
    def __init__(self, features, convert_to):
        self.features = features
        self.convert_to = convert_to

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df[self.features] = df[self.features].astype(self.convert_to)
        if self.convert_to == "str":
            df[self.features] = df[self.features].replace("nan", np.nan)
        return df


# Undersample
class Undersample(BaseEstimator, TransformerMixin):
    def __init__(self, n_max_per_level, random_state=42):
        self.n_max_per_level = n_max_per_level
        self.random_state = random_state
        self.b_sample = None
        self.b_all = None

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        return df

    def fit_transform(self, df, y=None, target="target"):
        self.b_all = df[target].mean()
        df = df.groupby(target).apply(lambda x: x.sample(min(self.n_max_per_level, x.shape[0]),
                                                         random_state=self.random_state)) \
            .reset_index(drop=True)\
            .sample(frac=1).reset_index(drop=True)
        self.b_sample = df[target].mean()
        return df


class FeatureEngineeringTitanic(BaseEstimator, TransformerMixin):
    def __init__(self, derive_deck=True, derive_familysize=True, derive_fare_pp=True):
        self.derive_deck = derive_deck
        self.derive_familysize = derive_familysize
        self.derive_fare_pp = derive_fare_pp

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        if self.derive_deck:
            df["deck"] = df["cabin"].str[:1]
        if self.derive_familysize:
            df["familysize"] = df["sibsp"].astype("int") + df["parch"].astype("int") + 1
        if self.derive_fare_pp:
            df["fare_pp"] = df.groupby("ticket")["fare"].transform("mean")
        return df


# Create sparse matrix
class CreateSparseMatrix(BaseEstimator, TransformerMixin):
    def __init__(self, metr=None, cate=None, df_ref=None):
        self.metr = metr
        self.cate = cate
        self.df_ref = df_ref
        self._d_categories = None

    def fit(self, df=None, y=None):
        if self.df_ref is None:
            self.df_ref = df
        if self.cate is not None and len(self.cate) > 0:
            self._d_categories = [self.df_ref[x].unique() for x in self.cate]
        return self

    def transform(self, df=None, y=None):
        if self.metr is not None and len(self.metr) > 0:
            m_metr = df[self.metr].to_sparse().to_coo()
        else:
            m_metr = None
        if self.cate is not None and len(self.cate) > 0:
            enc = OneHotEncoder(categories=self._d_categories)
            if len(self.cate) == 1:
                m_cate = enc.fit_transform(df[self.cate].reshape(-1, 1), y)
            else:
                m_cate = enc.fit_transform(df[self.cate], y)
        else:
            m_cate = None
        return hstack([m_metr, m_cate], format="csr")

