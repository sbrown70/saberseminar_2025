import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import polars as pl
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

import bambi as bmb
import arviz as az

COLOR_DICT = {
    "BLUE": "#2a5674",
    "RED": "#b13f64",
    "GOLD": "#e6b54a",
    "TEAL": "#4a9b8f",
    "CORAL": "#f47c65",
    "PURPLE": "#6a4a99",
    "ORANGE": "#f28c28",
    "GREEN": "#4a9b4a",
    "PINK": "#d47ca6"
}

def plot_measurement_errors_by_type(swing_df):
    COLOR_DICT = {
        "BLUE": "#2a5674",
        "RED": "#b13f64",
        "GOLD": "#e6b54a",
        "TEAL": "#4a9b8f",
        "CORAL": "#f47c65",
        "PURPLE": "#6a4a99",
        "ORANGE": "#f28c28",
        "GREEN": "#4a9b4a",
        "PINK": "#d47ca6"
    }
    
    colors = [COLOR_DICT[COLOR] for COLOR in COLOR_DICT.keys()]
    _ = (
        swing_df
        .filter(~pl.col("pitch_type").is_in(["CS","FO","FA","HC","KN","SV"]))
        .select(["pitch_type","swing_path_tilt_error"])
        .group_by(["pitch_type"])
        .mean()
        .sort(by="swing_path_tilt_error",descending=True)
    )
    
    fig,axs=plt.subplots(len(_["pitch_type"].unique()),3,figsize=(9,12))
    
    for i, pitch_type in enumerate(_["pitch_type"]):
        ax=axs[i,0]
        sns.kdeplot(
            data=swing_df.filter(pl.col("pitch_type")==pitch_type),
            x="swing_path_tilt_error",
            color=f"C{i}",
            label=pitch_type,
            ax=ax,
            fill=True,
            alpha=0.2,
        )
    
        ax.axvline(0,color="black",linestyle="--",alpha=0.5,lw=1.5)
        
        ax.set(
            # title="Tilt Error" if i==0 else "",
            xlabel="Tilt Error" if i==len(_["pitch_type"].unique())-1 else "",
            xlim=(-25,25),
            ylabel=pitch_type,
            yticks=[],
            xticks=[-20,0,20],
            xticklabels=[] if i!=len(_["pitch_type"].unique())-1 else ["Over","","Under"],
        )
    
        ax.spines[["left","right","top"]].set_visible(False)
    
    for i, pitch_type in enumerate(_["pitch_type"]):
        ax=axs[i,1]
        sns.kdeplot(
            data=swing_df.filter(pl.col("pitch_type")==pitch_type),
            x="int_y_error",
            color=f"C{i}",
            label=pitch_type,
            ax=ax,
            fill=True,
            alpha=0.2,
        )
    
        ax.axvline(0,color="black",linestyle="--",alpha=0.5,lw=1.5)
        
        ax.set(
            # title="Y Intercept Error" if i==0 else "",
            xlabel="Y Intercept Error" if i==len(_["pitch_type"].unique())-1 else "",
            xlim=(-25,25),
            ylabel="",
            yticks=[],
            xticks=[-20,0,20],
            xticklabels=[] if i!=len(_["pitch_type"].unique())-1 else ["Behind","","Ahead"],
        )
    
        ax.spines[["left","right","top"]].set_visible(False)
    
    for i, pitch_type in enumerate(_["pitch_type"]):
        ax=axs[i,2]
        sns.kdeplot(
            data=swing_df.filter(pl.col("pitch_type")==pitch_type),
            x="bat_speed_error",
            color=f"C{i}",
            label=pitch_type,
            ax=ax,
            fill=True,
            alpha=0.2,
        )
    
        ax.axvline(0,color="black",linestyle="--",alpha=0.5,lw=1.5)
        
        ax.set(
            # title="Bat Speed Error" if i==0 else "",
            xlabel="Bat Speed Error" if i==len(_["pitch_type"].unique())-1 else "",
            xlim=(-25,25),
            ylabel="",
            yticks=[],
            xticks=[-20,0,20],
            xticklabels=[] if i!=len(_["pitch_type"].unique())-1 else ["Slower","","Faster"],
        )
    
        ax.spines[["left","right","top"]].set_visible(False)
    
    plt.suptitle("""Bat Tracking Measurements vs Expected by Pitch Type
    Expected based on batter ID, pitch type, velo, and location.""")
    
    plt.show()

def plot_over_under(
    swing_df,
    detail_level=1000,
    miss_cols=["under","behind"],
    title_text="""Over / Under Probability is the probability a batter would expect a pitch with more or less
vertical movement (including gravity) than what the pitch actually exhibited. It leverages
expected pitch type probabilities given the pitcher's arm angle and the initial trajectory of
the pitch. Expected vertical movement with gravity for each probable pitch type is based on 
arm angle and extension.

Over / Under Frequency is the frequency at which a batter is over or under a pitch with the
corresponding over / under probability.

Calibration results below exclude fastballs.""",
): 
    COLOR_DICT = {
        "BLUE": "#2a5674",
        "RED": "#b13f64",
        "GOLD": "#e6b54a",
        "TEAL": "#4a9b8f",
        "CORAL": "#f47c65",
        "PURPLE": "#6a4a99",
        "ORANGE": "#f28c28",
        "GREEN": "#4a9b4a",
        "PINK": "#d47ca6"
    }
    
    colors = [COLOR_DICT[COLOR] for COLOR in COLOR_DICT.keys()]
    fig, axs = plt.subplots(1, 2, facecolor="#eeeeee", figsize=(9,6))
    
    for i,miss in enumerate(miss_cols):
        ax = axs[i]
        plot_data = (
            swing_df
            # .filter(pl.col("n_thruorder_pitcher")==1)
            .filter(pl.col(f"{miss}_probability")<0.9875)
            .filter(pl.col(f"{miss}_probability")>(1-0.9875))
            # .filter(pl.col("pitch_group").is_in([
            #     "curve",
            #     "slider",
            #     "offspeed",
            # ]))
            .filter(~(pl.col("pitch_group")=="fastball"))
            .with_columns(
                under_probability=pl.col(f"{miss}_probability").mul(detail_level).cast(pl.Int32).mul(1/(detail_level/100)),
                under=pl.col(miss).mul(100),
            )
            .group_by(["under_probability"])
            .agg(
                under_frequency=pl.col("under").mean(),
                n=pl.col("under").count(),
            )
        )
        
        corr_matrix = plot_data.to_pandas().drop(columns=["n"]).corr()
        under_corr = corr_matrix.iloc[0, 1]
    
        sns.regplot(
            data=plot_data,
            x="under_probability",
            y="under_frequency",
            color=COLOR_DICT["RED"] if i==0 else COLOR_DICT["BLUE"],
            ax=ax,
            scatter=False,
        )
        
        sns.scatterplot(
            data=plot_data,
            x="under_probability",
            y="under_frequency",
            color=COLOR_DICT["RED"] if i==0 else COLOR_DICT["BLUE"],
            ax=ax,
            size="n",
            sizes=(5, 100),
            alpha=0.5,
        )
        ax.get_legend().remove()
        ax.set(
            xlim=(0, 105),
            ylim=(0, 105),
            facecolor="#eeeeee",
            xlabel=f"{miss} probability",
            ylabel=f"{miss} frequency",
        )
        ax.set_aspect("equal")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["bottom"].set_bounds(0,100)
        ax.plot(
            [0, 100],
            [0, 100],
            linestyle="--",
            color="black",
            alpha=0.5,
            lw=1.5,
        )
        
        ax.text(
            x=0.05,
            y=0.95,
            s=f"R: {under_corr:.3f}", 
            transform=ax.transAxes,
            fontsize=10, 
            va="top",
        )
    
    title_base = fig.text(
        s=title_text,
        x=0.0,
        ha="left",
        y=0.9,
        va="bottom",
        fontsize=14,
    )
    
    plt.tight_layout()
    plt.show()

def plot_study_results_gam(
    study_results,
    title_text="",
    subtitle_text="",
    ylimits=[],
):
    COLOR_DICT = {
        "BLUE": "#2a5674",
        "RED": "#b13f64",
        "GOLD": "#e6b54a",
        "TEAL": "#4a9b8f",
        "CORAL": "#f47c65",
        "PURPLE": "#6a4a99",
        "ORANGE": "#f28c28",
        "GREEN": "#4a9b4a",
        "PINK": "#d47ca6"
    }
    
    colors = [COLOR_DICT[COLOR] for COLOR in COLOR_DICT.keys()]
    fig, ax = plt.subplots(facecolor="#eeeeee")
    
    for i, x_var in enumerate([
        col for col in study_results.columns if col.startswith("params_")
    ]):
        
        x_values = study_results[x_var]
        x_min, x_max = x_values.min(), x_values.max()
        x_normalized = (x_values - x_min) / (x_max - x_min) if x_max != x_min else x_values * 0
        
        plot_data = study_results.copy()
        plot_data[f'{x_var}_normalized'] = x_normalized
        
        sns.regplot(
            data=plot_data,
            x=f'{x_var}_normalized',
            y="value",
            color="#eeeeee",
            ax=ax,
            lowess=True,
            scatter=False,
            line_kws=dict(lw=8),
        )
        
        sns.regplot(
            data=plot_data,
            x=f'{x_var}_normalized',
            y="value",
            color=colors[i],
            ax=ax,
            label=x_var.split("params_")[1],
            lowess=True,
            scatter=False,
            line_kws=dict(lw=5),
        )
        ax.scatter(
            x=x_normalized,
            y=study_results["value"],
            color=colors[i],
            label=None,
            clip_on=False,
            alpha=0.25,
        )
    
    ax.set(
        ylabel="Swing Miss Model Performance\n(AUC)",
        xlabel="Parameter Value (Normalized: 0 = Min, 1 = Max)",
        facecolor="#eeeeee",
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.05), ncols=2, frameon=False, title="Input Variable")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1)
    if len(ylimits)>0:
        ax.set_ylim(ylimits)
    
    # text_orig = ax.text(
    #     s="""Modeled probability of swing given pitch type usage rates, ball location in 
    # batter's field of vision at three points in time following release, and change 
    # in visual angle of the diameter of the ball from release to decision point,
    # with random jitter applied to locations to capture visual measurement error.
    # The amount of jitter at each point was varied using Optuna to optimize AUC.""",
    #     x=0, ha="left",
    #     y=1.15, va="bottom",
    #     transform=fig.transFigure,
    #     fontsize=12,
    #     style="italic",
    #     alpha=0.75,
    # )
    
    # text_orig = ax.annotate(
    #     xy=(0, 1.25),
    #     ha="left",
    #     va="bottom",
    #     xycoords=text_orig,
    #     fontsize=15,
    #     weight="bold",
    #     text="""Swing Probability Model Performance as a Function of 
    # Visual Angle Uncertainty (Visual Measurement Error)""",
    # )
    
    text_orig = ax.text(
        s=subtitle_text,
        x=0, ha="left",
        y=1.15, va="bottom",
        transform=fig.transFigure,
        fontsize=12,
        style="italic",
        alpha=0.75,
    )
    
    text_orig = ax.annotate(
        xy=(0, 1.25),
        ha="left",
        va="bottom",
        xycoords=text_orig,
        fontsize=15,
        weight="bold",
        text=title_text,
    )
    
    plt.show()

def shohei_plots(swing_df):
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon
    plot_data=swing_df.filter(pl.col("batter")==660271).filter(pl.col("well_hit")==1)
    
    fig,ax=plt.subplots(facecolor="#eeeeee")
    
    sns.scatterplot(
        ax=ax,
        data=plot_data,
        x="plate_x",
        y="plate_z",
        hue="swing_path_tilt",
        palette="Blues",
        edgecolor="black",
        s=75,
        legend=False,
    )
    
    strikezone = patches.Rectangle(
        (-0.83, 1.5),
        1.66,
        2.0,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
    )
    ax.add_patch(strikezone)
    
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    
    ax.set(
        facecolor="#eeeeee",
        xlabel="",
        ylabel="",
        xticks=[],
        yticks=[],
        title="""2024 Shohei Ohtani\nTilt vs Location\nHard-Hit Balls in Play"""
    )
    
    ax.set_aspect("equal", adjustable="box")
    
    norm = plt.Normalize(plot_data["swing_path_tilt"].min(), plot_data["swing_path_tilt"].max())
    sm = plt.cm.ScalarMappable(cmap="Blues", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm,ax=ax,label="Swing Path Tilt (degrees)",location="left")
    
    ax.text(
        x=0,y=1,ha="center",
        transform=ax.transData,
        s="swing_path_tilt ~ 1 + (plate_z|batter/pitch_group)",
        fontsize=10,
        alpha=0.5,
        style="italic",
    )
    
    plt.show()
    
    ################ Velocity vs Int
    
    plot_data=(
        swing_df
        .filter(pl.col("batter")==660271)
        .filter(pl.col("well_hit")==1)
        .with_columns(
            pl.col("int_y").mul(1/12).sub(2).alias("int_y_feet"),
        )
        .filter(pl.col("release_speed").is_between(80,100))
    )
    
    fig,ax=plt.subplots(facecolor="#eeeeee")
    
    sns.scatterplot(
        ax=ax,
        data=plot_data,
        x="plate_x",
        y="int_y_feet",
        hue="release_speed",
        palette="Reds",
        edgecolor="black",
        s=75,
        legend=False,
    )
    
    ax.spines[["top","right","bottom","left"]].set_visible(False)
    
    ax.set(
        facecolor="#eeeeee",
        xlabel="",
        ylabel="",
        xticks=[],
        yticks=[],
        title="""2024 Shohei Ohtani\nVelocity vs Intercept Point\nHard-Hit Balls in Play"""
    )
    
    ax.set_aspect("equal", adjustable="box")  # Equal scaling for both axes
    
    norm = plt.Normalize(plot_data["release_speed"].min(), plot_data["release_speed"].max())
    sm = plt.cm.ScalarMappable(cmap="Reds", norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Pitch Velocity (mph)")
    
    home_plate = Polygon([
        (-0.7083, 0),  # top left corner
        (0.7083, 0),   # top right corner
        (0.7083, -0.7083),   # bottom right point
        (0.0, -1.4167),         # bottom tip (toward batter)
        (-0.7083, -0.7083),  # bottom left point
    ], closed=True, edgecolor="black", facecolor="none", linewidth=2)
    
    ax.add_patch(home_plate)
    
    ax.text(
        x=0,y=-2,ha="center",
        transform=ax.transData,
        s="int_y ~ 1 + (tf+plate_x_bat_flip|batter/pitch_group)",
        fontsize=10,
        alpha=0.5,
        style="italic",
    )
    
    plt.show()

def equation_plots():
    under_prob_eq = (
        r"$P_{\text{Under}} \;=\; \sum_{i=0}^{N} \; P(\text{Cluster}_i) \; \cdot \; "
        r"\mathbf{1}\!\left( \widehat{\text{VERT}}_i \;>\; \text{VERT}_{\text{obs}} \right)$"
    )
    
    ahead_prob_eq = (
        r"$P_{\text{Ahead}} \;=\; \sum_{i=0}^{N} \; P(\text{Cluster}_i) \; \cdot \; "
        r"\mathbf{1}\!\left( \widehat{\text{VELO}}_i \;<\; \text{VELO}_{\text{obs}} \right)$"
    )
    
    fig, ax = plt.subplots(facecolor="#eeeeee")
    
    ax.axis("off")
    
    ax.text(0.5, 0.7, under_prob_eq, fontsize=16, ha="center", va="center")
    ax.text(0.5, 0.3, ahead_prob_eq, fontsize=16, ha="center", va="center")
    
    plt.show()

def whiff_well_hit_validation_plot(swing_df):
    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(1,2,facecolor="#eeeeee")
    
    COLORS = [COLOR_DICT["PURPLE"], COLOR_DICT["TEAL"]]
    
    format_dictionary = {
        "swing_path_tilt_error": {
            "title": "Tilt Error",
            "xlim": (-18, 18),
            "xticks": [-20 / 25 * 18, 0, 20 / 25 * 18],
            "xticklabels": ["Under", "", "Over"],
        },
        "int_y_error": {
            "title": "Contact Point Error",
            "xlim": (-30, 30),
            "xticks": [-20 / 25 * 30, 0, 20 / 25 * 30],
            "xticklabels": ["Behind", "", "Ahead"],
        },
        "bat_speed_error": {
            "title": "Bat Speed Error",
            "xlim": (-20, 20),
            "xticks": [-20 / 25 * 20, 0, 20 / 25 * 20],
            "xticklabels": ["Slower", "", "Faster"],
        },
    }
    
    for j,measure in enumerate(["swing_path_tilt_error","int_y_error"]):
        ax=axs[j]
        
        for i, pitch_group in enumerate(["is_whiff","well_hit"]):
            _plot_data = swing_df.filter(pl.col(pitch_group) == 1).filter(pl.col("pitch_type")=="SL").sample(250)
            if j==0:
                y_vals = _plot_data[measure].to_pandas().mul(-1).to_numpy()
            else:
                y_vals = _plot_data[measure].to_pandas().mul(1).to_numpy()
                    
            kde = gaussian_kde(y_vals)
            density = kde(y_vals)
            density_scaled = density / density.max()
            jitter = np.random.uniform(-1, 1, size=len(y_vals)) * density_scaled * 0.2
            _x = i + jitter
        
            sns.scatterplot(
                ax=ax,
                x=_x,
                y=y_vals,
                color=COLORS[i],
                s=65,
                alpha=0.25,
                edgecolor=COLORS[i],
                clip_on=False,
            )
        
            box_whisker_props={"linewidth":1.5, "color":"#a9a9a9","alpha":1}
            box_whisker_props={"linewidth":2.0, "color":"black","alpha":0.75}
            ax.boxplot(
                x=y_vals,
                positions=[0.0 + i],
                vert=True,
                manage_ticks=False,
                showfliers=False,
                showcaps=False,
                boxprops=box_whisker_props,
                whiskerprops=box_whisker_props,
                medianprops=box_whisker_props,
            )
            ax.set(
                facecolor="#eeeeee",
                title="",
                ylabel=format_dictionary[measure]["title"],
                ylim=format_dictionary[measure]["xlim"],
                yticks=format_dictionary[measure]["xticks"],
                yticklabels=format_dictionary[measure]["xticklabels"],
                xlabel="",
                xticks=[],
                xlim=[-0.35, 1.5],
            )
            ax.spines[["top","right","bottom","left"]].set_visible(False)
            if j==0:
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_bounds(np.min(format_dictionary[measure]["xticks"]), np.max(format_dictionary[measure]["xticks"]))
            else:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_bounds(np.min(format_dictionary[measure]["xticks"]), np.max(format_dictionary[measure]["xticks"]))
    
    breaking_balls_init = fig.text(
        x=0.05,y=1.1,ha="left",va="top",
        s="""Batters have significantly more tilt
    and contact point error on whiffs""",
        fontsize=12,
        weight="bold",
        color=COLORS[0],
    )
    
    fastballs_init = fig.text(
        x=0.875,y=0.0,ha="right",va="top",
        s="""Than on well-hit balls in play""",
        fontsize=12,
        weight="bold",
        color=COLORS[1],
    )
    
    axs[0].annotate(
        "",
        xy=(0, 15),
        xycoords="data",
        xytext=(0.30, 1.05),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.8,connectionstyle="arc3,rad=-0.3",),
        clip_on=False,
    )
    
    axs[1].annotate(
        "",
        xy=(0, 30),
        xycoords="data",
        xytext=(0.30, 1.05),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.8,connectionstyle="arc3,rad=0.3",),
        clip_on=False,
    )
    
    axs[0].annotate(
        "",
        xy=(1,-8),
        xycoords="data",
        xytext=(0.675, .125),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.8,connectionstyle="arc3,rad=-0.3",),
        clip_on=False,
    )
    
    axs[1].annotate(
        "",
        xy=(1, -25),
        xycoords="data",
        xytext=(0.675, .125),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.8,connectionstyle="arc3,rad=0.1",),
        clip_on=False,
    )
    
    axs[0].axvline(1.6,clip_on=False,color="black",lw=0.75,alpha=0.5,)
    
    
    plt.show()

def fastball_vs_breaking_ball_plot(swing_df):
    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(1,2,facecolor="#eeeeee")
    
    COLORS = [COLOR_DICT["BLUE"], COLOR_DICT["RED"]]
    
    format_dictionary = {
        "swing_path_tilt_error": {
            "title": "Tilt Error",
            "xlim": (-18, 18),
            "xticks": [-20 / 25 * 18, 0, 20 / 25 * 18],
            "xticklabels": ["Under", "", "Over"],
        },
        "int_y_error": {
            "title": "Contact Point Error",
            "xlim": (-30, 30),
            "xticks": [-20 / 25 * 30, 0, 20 / 25 * 30],
            "xticklabels": ["Behind", "", "Ahead"],
        },
        "bat_speed_error": {
            "title": "Bat Speed Error",
            "xlim": (-20, 20),
            "xticks": [-20 / 25 * 20, 0, 20 / 25 * 20],
            "xticklabels": ["Slower", "", "Faster"],
        },
    }
    
    for j,measure in enumerate(["swing_path_tilt_error","int_y_error"]):
        ax=axs[j]
        
        # measure = "swing_path_tilt_error"
        
        for i, pitch_group in enumerate(["breaking", "fastball"]):
            _plot_data = swing_df.filter(pl.col("simple_pitch_group") == pitch_group).sample(250)
            if j==0:
                y_vals = _plot_data[measure].to_pandas().mul(-1).to_numpy()
            else:
                y_vals = _plot_data[measure].to_pandas().mul(1).to_numpy()
                    
            kde = gaussian_kde(y_vals)
            density = kde(y_vals)
            density_scaled = density / density.max()
            jitter = np.random.uniform(-1, 1, size=len(y_vals)) * density_scaled * 0.2
            _x = i + jitter
        
            sns.scatterplot(
                ax=ax,
                x=_x,
                y=y_vals,
                color=COLORS[i],
                s=65,
                alpha=0.25,
                edgecolor=COLORS[i],
            )
        
            box_whisker_props={"linewidth":1.5, "color":"#a9a9a9","alpha":1}
            box_whisker_props={"linewidth":2.0, "color":"black","alpha":0.75}
            ax.boxplot(
                x=y_vals,
                positions=[0.0 + i],
                vert=True,
                manage_ticks=False,
                showfliers=False,
                showcaps=False,
                boxprops=box_whisker_props,
                whiskerprops=box_whisker_props,
                medianprops=box_whisker_props,
            )
            ax.set(
                facecolor="#eeeeee",
                title="",
                ylabel=format_dictionary[measure]["title"],
                ylim=format_dictionary[measure]["xlim"],
                yticks=format_dictionary[measure]["xticks"],
                yticklabels=format_dictionary[measure]["xticklabels"],
                xlabel="",
                xticks=[],
                xlim=[-0.35, 1.5],
            )
            ax.spines[["top","right","bottom","left"]].set_visible(False)
            if j==0:
                ax.spines["left"].set_visible(True)
                ax.spines["left"].set_bounds(np.min(format_dictionary[measure]["xticks"]), np.max(format_dictionary[measure]["xticks"]))
            else:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(True)
                ax.spines["right"].set_bounds(np.min(format_dictionary[measure]["xticks"]), np.max(format_dictionary[measure]["xticks"]))
    
    breaking_balls_init = fig.text(
        x=0.05,y=1.05,ha="left",va="top",
        s="""Batters' swings on breaking balls
    are often over and ahead of the pitch""",
        fontsize=12,
        weight="bold",
        color=COLORS[0],
    )
    
    fastballs_init = fig.text(
        x=0.875,y=0.0,ha="right",va="top",
        s="""While being under and behind
    the pitch on swings vs fastballs""",
        fontsize=12,
        weight="bold",
        color=COLORS[1],
    )
    
    axs[0].annotate(
        "",
        xy=(0, 12),
        xycoords="data",
        xytext=(0.30, 1.05),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.8,connectionstyle="arc3,rad=-0.3",),
        clip_on=False,
    )
    
    axs[1].annotate(
        "",
        xy=(0, 29),
        xycoords="data",
        xytext=(0.30, 1.05),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[0], lw=1.8,connectionstyle="arc3,rad=0.3",),
        clip_on=False,
    )
    
    axs[0].annotate(
        "",
        xy=(1,-12),
        xycoords="data",
        xytext=(0.675, .125),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.8,connectionstyle="arc3,rad=-0.15",),
        clip_on=False,
    )
    
    axs[1].annotate(
        "",
        xy=(1, -27),
        xycoords="data",
        xytext=(0.675, .125),
        textcoords="figure fraction",
        arrowprops=dict(arrowstyle="->", color=COLORS[1], lw=1.8,connectionstyle="arc3,rad=0.1",),
        clip_on=False,
    )
    
    axs[0].axvline(1.6,clip_on=False,color="black",lw=0.75,alpha=0.5,)
    
    plt.show()

def plot_feature_importances(
    predicted_pitch_types_model_name="initial_pitch_cluster_catboost",
    title="Pitch Type Classification",
):
    import catboost as cb
    BLUE = "#2a5674"
    RED = "#b13f64"
    
    model_used = cb.CatBoostClassifier(loss_function="MultiClass")
    model_used.load_model(f"{predicted_pitch_types_model_name}.cbm",format="cbm")
        
    x = model_used.feature_names_
    y = model_used.get_feature_importance()
    
    feature_importance = model_used.get_feature_importance()
    feature_names = model_used.feature_names_
    sorted_indices = feature_importance.argsort()[::-1]
    sorted_feature_importance = feature_importance[sorted_indices]
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(facecolor = "#eeeeee")
    
    bar_colors = [RED] * len(sorted_feature_importance)
    
    sns.barplot(
        x = sorted_feature_importance,
        y = sorted_feature_names,
        ax = ax,
        palette = bar_colors,
    )
    
    ax.set_facecolor("#eeeeee")
    sns.despine(ax = ax)
    
    ax.set_xlabel("feature importance")
    ax.set_title(title)
    
    plt.show()

def generic_results_plots(
    results_df,
    param_cols=3,
    y_cols=["value","ahead_auc","under_auc","mean_cluster_auc","total_auc","ahead_brier","under_brier","total_brier"],
    title_string="Pitch Cluster Prediction Model Optimization Results",    
):

    COLOR_DICT = {
        "BLUE": "#2a5674",
        "RED": "#b13f64",
        "TEAL": "#4a9b8f",
        "CORAL": "#f47c65",
        "PURPLE": "#6a4a99",
        "ORANGE": "#f28c28",
        "GREEN": "#4a9b4a",
        "PINK": "#d47ca6",
        "MAROON": "#8b2439",
        "NAVY": "#1a237e",
        "LIME": "#b6e880",
        "CYAN": "#00bcd4",
        "BROWN": "#795548",
        "MAGENTA": "#e040fb",
        "GOLD": "#e6b54a",
    }
    
    colors = [COLOR_DICT[color] for color in COLOR_DICT.keys()]

    import math 
    
    params_dict = {
        "decision_point_location":{
            "title":"Decision Point Time",
            "xlimits":[40,250],
            "xbounds":[50,250],
            "xticks":np.arange(50,300,50),
            "importance":0.35,
            "rank":"1 (tie)",
        },
        "visual_angle_change_cuts":{
            "title":"Velocity Precision",
            "xlimits":[1,40],
            "xbounds":[2,40],
            # "xticks":np.arange(2,26,6),
            "xticks":[2,10,20,30,40],
            "importance":0.35,
            "rank":"1 (tie)",
        },
        "release_angle_visual_cuts":{
            "title":"Release Angle Precision",
            "xlimits":[1,100],
            "xbounds":[2,100],
            "xticks":[2,20,40,60,80,100],
            "importance":0.22,
            "rank":3,
        },
        "arm_angle_cuts":{
            "title":"Arm Angle Precision",
            "xlimits":[1,20],
            "xbounds":[2,20],
            "xticks":np.arange(2,26,6),
            "importance":0.06,
            "rank":4,
        },
        "tot_distance_cuts":{
            "title":"Trajectory Length Precision",
            "xlimits":[18,100],
            "xbounds":[20,100],
            "xticks":np.arange(20,120,20),
            "importance":0.02,
            "rank":5,
        },
        "arc_depth_cuts":{
            "title":"Arc Depth Precision",
            "xlimits":[18,100],
            "xbounds":[20,100],
            "xticks":np.arange(20,120,20),
            "importance":0.01,
            "rank":6,
        },
    }
    params = [col.split("params_")[1] for col in [_ for _ in results_df.columns if _.startswith("params_")]]
    for y_column in [_y_col for _y_col in y_cols if _y_col in results_df.columns]:
        
        fig,axs=plt.subplots(math.ceil(len(params)/3),param_cols,facecolor="#eeeeee",sharey=True)
        flat_ax=axs.flatten()
        
        for i, param in enumerate(params):
                
            ax=flat_ax[i]
            
            sns.regplot(
                data=results_df,
                x=f"params_{param}",
                y=y_column,
                color="#eeeeee",
                ax=ax,
                lowess=True,
                scatter=False,
                line_kws=dict(lw=8),
            )
        
            sns.regplot(
                data=results_df,
                x=f"params_{param}",
                y=y_column,
                color=colors[i],
                ax=ax,
                lowess=True,
                scatter=False,
                line_kws=dict(lw=5),
            )
        
            ax.scatter(
                x=results_df[f"params_{param}"],
                y=results_df[y_column],
                color=colors[i],
                label=None,
                clip_on=False,
                alpha=0.25,
            )
        
            ax.set(
                ylabel=y_column,
                xlabel="",
                facecolor="#eeeeee",
            )
        
            ax.set_title(
                params_dict[param]["title"],
                color=colors[i],
                fontsize=9,
            )
            ax.spines[["top", "right"]].set_visible(False)
    
        plt.suptitle(title_string)
        plt.tight_layout()
        plt.show()