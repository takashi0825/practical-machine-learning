import pandas as pd
import sklearn.metrics
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

def evaluate_regression(t, y, yyplot=True, yyplot_png=True, yyplot_svg=False, show_scores=True, yyplot_filename="yyplot"):
    # Error check
    if len(t) != len(y):
        raise ValueError("len(t) != len(y) : " + str(len(t)) + " != " + str(len(y)))

    # Stats
    output = {}
    output["N"] = len(t)
    output["t_min"] = np.min(t)
    output["t_med"] = np.median(t)
    output["t_mean"] = np.mean(t)
    output["t_max"] = np.max(t)
    output["t_var"] = np.var(t)
    output["t_std"] = np.std(t)
    output["y_min"] = np.min(y)
    output["y_med"] = np.median(y)
    output["y_mean"] = np.mean(y)
    output["y_max"] = np.max(y)
    output["y_var"] = np.var(y)
    output["y_std"] = np.std(y)

    # Metrics
    output["R2"] = sklearn.metrics.r2_score(t, y)
    output["MSE"] = sklearn.metrics.mean_squared_error(t, y)
    output["MAE"] = sklearn.metrics.mean_absolute_error(t, y)
    t2 = t[t > 0]
    y2 = y[t > 0]
    output["MAPE"] = np.mean(np.abs(t2 - y2) / np.abs(t2))
    output["WAPE"] = np.sum(np.abs(t - y)) / np.sum(t)

    # Output dataframe
    df_output = pd.DataFrame(output.values(), index=output.keys(), columns =["score"])

    # Show scores
    if show_scores:
        pd.options.display.float_format = '{:.3f}'.format
        display(df_output)
        pd.options.display.float_format = None

    # yyplot
    if yyplot:
        plt.figure(figsize=(8,8))
        lim_min = np.min([np.min(t), np.min(y)])
        lim_max = np.max([np.max(t), np.max(y)])
        plt.xlabel("Prediction")
        plt.ylabel("Actual")
        plt.xlim(lim_min, lim_max)
        plt.ylim(lim_min, lim_max)
        plt.scatter(y, t, s=6)
        plt.plot([lim_min,lim_max], [lim_min,lim_max], linestyle=":", color="black")
        if yyplot_png:
            plt.savefig(yyplot_filename + ".png", dpi=300, bbox_inches='tight', pad_inches=0)
        if yyplot_svg:
            plt.savefig(yyplot_filename + ".svg", bbox_inches='tight', pad_inches=0)
        plt.show()

    return df_output
