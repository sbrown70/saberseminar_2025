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

def flip_batter_cols(
    df,
    flip_cols=["release_pos_x","plate_x","pfx_x"],
):
    for flip_col in flip_cols:
        df = df.with_columns(
            pl.when(pl.col("stand")=="L").then(pl.col(flip_col)*-1).otherwise(pl.col(flip_col)).alias(f"{flip_col}_bat_flip"),
        )

    return df

def add_event_cols(df):
    df = df.with_columns(
        pl.lit(np.where(
            df["description"] != "hit_into_play",
            "--",
            np.where(
                df["events"].is_in(["single","double","triple","home_run"]),
                df["events"],
                "bip_out"
            )
        )).alias("bip_event")
    )
    
    descriptions_init = list(df["description"].unique())
    swing_descriptions = [desc for desc in descriptions_init if "foul" in desc or "bunt" in desc or "swing" in desc or "hit_into" in desc]
    strike_descriptions = [desc for desc in descriptions_init if "foul" in desc or "strike" in desc or "missed_bunt" in desc]
    ball_descriptions = ["ball","blocked_ball"]
    whiff_descrptions = [desc for desc in descriptions_init if "swinging_strike" in desc or "foul_tip" in desc]
    
    df=df.with_columns(
        pl.col("description").is_in(swing_descriptions).cast(pl.Int32).alias("is_swing"),
        pl.col("description").is_in(whiff_descrptions).cast(pl.Int32).alias("is_whiff"),
        pl.col("description").is_in(["hit_into_play"]).cast(pl.Int32).alias("is_in_play"),
    )
    
    df=df.with_columns(
        (1-pl.col("is_swing")).alias("is_take"),
        (pl.col("is_swing")-pl.col("is_whiff")-pl.col("is_in_play")).alias("is_foul"),
        pl.col("bip_event").is_in(["bip_out"]).cast(pl.Int32).alias("is_out")
    )
    
    df=df.with_columns(
        pl.col("description").is_in(strike_descriptions).cast(pl.Int32).alias("is_strike"),
        pl.col("description").is_in(ball_descriptions).cast(pl.Int32).alias("is_ball")
    )
    
    df=df.with_columns(
        pl.lit(np.where(df["is_take"]==0,0,df["is_strike"])).alias("is_called_strike"),
        pl.lit(np.where(df["is_take"]==0,0,df["is_ball"])).alias("is_called_ball")
    )
    
    df = df.with_columns([
        pl.col("strikes").shift(-1).fill_null(0).alias("post_strikes"),
        pl.col("balls").shift(-1).fill_null(0).alias("post_balls"),
    ])
    
    df = df.with_columns(
        pl.when(
            (pl.col("strikes") == 2) &
            (pl.col("post_strikes") > 0) &
            (pl.col("is_foul") == 1)
        )
        .then(0)
        .otherwise(pl.col("is_strike"))
        .alias("is_strike")
    )
    
    
    df=df.with_columns(pl.lit(np.where(
        df["description"] == "called_strike",
        "called_strike",
        np.where(
            df["description"]=="ball",
            "called_ball",
            np.where(df["description"]=="hit_by_pitch","hit_by_pitch",None)
        )
    )).alias("take_event"))
    
    df=df.with_columns(pl.lit(np.where(
        df["is_swing"]!=1,
        None,
        np.where(
            df["description"].str.contains("foul"),
            "foul",
            np.where(
                df["description"].str.contains("swinging_strike"),
                "whiff",
                np.where(
                    df["description"].str.contains("missed_bunt"),
                    "whiff",
                    "hit_into_play"
                )
            )
        )
    )).alias("swing_event"))
    
    df=df.with_columns(pl.lit(np.where(df["is_swing"]==1,"swing","take")).alias("swing_or_take"))

    return df


def find_competitive_swings(df):
    for col in ["bat_speed","swing_length","launch_speed","launch_angle"]:
        df = df.with_columns(pl.col(col).cast(pl.Float64).alias(col))
    
    lf = df.lazy()
    
    lf = lf.with_columns(pl.col("bat_speed").quantile(0.1,"nearest").over(["batter", "game_year", "stand"]).alias("10pctile_bat_speed"))
    
    lf = lf.with_columns([
        pl.when((pl.col("bat_speed") > pl.col("10pctile_bat_speed")) |
                ((pl.col("bat_speed") >= 60) & (pl.col("launch_speed") >= 90)))
        .then(pl.col("bat_speed"))
        .otherwise(None)
        .alias("competitive_bat_speed")
    ])
    
    lf = lf.with_columns([
        pl.when((pl.col("bat_speed") > pl.col("10pctile_bat_speed")) |
                ((pl.col("bat_speed") >= 60) & (pl.col("launch_speed") >= 90)))
        .then(pl.col("swing_length"))
        .otherwise(None)
        .alias("competitive_swing_length")
    ])
    
    for col in ["bat_speed", "swing_length"]:
        lf = lf.with_columns([
            pl.col(col).alias(f"{col}_orig"),
            pl.when((pl.col("bat_speed") > pl.col("10pctile_bat_speed")) |
                    ((pl.col("bat_speed") >= 60) & (pl.col("launch_speed") >= 90)))
            .then(pl.col(col))
            .otherwise(None)
            .alias(col)
        ])
    
        lf = lf.with_columns(pl.col(col).mean().over(["batter", "game_year", "stand"]).alias(f"average_{col}"))
    
    df = lf.collect()

    return df

def add_pitch_group(df):
    df = df.with_columns(
        pl.when(pl.col("pitch_type").is_in(["FA","SI","FF"]))
        .then(pl.lit("fastball"))
        .otherwise(
            pl.when(pl.col("pitch_type")=="FC")
            .then(
                pl.when(pl.col("release_speed")>=92.5).then(pl.lit("slider")).otherwise(pl.lit("slider"))
            )
            .otherwise(
                pl.when(pl.col("pitch_type").is_in(["ST","SL","SV"])).then(pl.lit("slider"))
                .otherwise(
                    pl.when(pl.col("pitch_type").is_in(["CU","KC","CS"])).then(pl.lit("curve")).otherwise(pl.lit("offspeed"))
                )
            )
        ).alias("pitch_group")
    )
    return df

def add_flight_metrics_and_squared_up_pct(df):
    df = (
        df
        .with_columns(
            y0 = pl.lit(50)
        )
        .with_columns(
            (60.5-pl.col("release_extension")).alias("yR")
        )
        .with_columns(
            ((-pl.col("vy0") - np.sqrt(pl.col("vy0")**2 - 2 * pl.col("ay") * (50 - pl.col("yR")))) / pl.col("ay")).alias("tR")
        )
        .with_columns(
            (pl.col("vx0")+pl.col("ax")*pl.col("tR")).alias("vxR")
        )
        .with_columns(
            (pl.col("vy0")+pl.col("ay")*pl.col("tR")).alias("vyR")
        )
        .with_columns(
            (pl.col("vz0")+pl.col("az")*pl.col("tR")).alias("vzR")
        )
        .with_columns(
            ((-pl.col("vyR") - np.sqrt(pl.col("vyR")**2 - 2 * pl.col("ay") * (pl.col("yR") - 17/12))) / pl.col("ay")).alias("tf")
        )
    )
    
    for col in df.columns:
        try:
            df = df.with_columns(
                pl.when(pl.col(col).is_infinite())
                .then(None)
                .otherwise(pl.col(col)
                ).alias(col)
            )
        except:
            continue
    
    df=df.with_columns(
        vxf=pl.col("vx0") + pl.col("ax") * pl.col("tf"),
        vyf=pl.col("vy0") + pl.col("ay") * pl.col("tf"),
        vzf=pl.col("vz0") + pl.col("az") * pl.col("tf"),
    )
    
    df=df.with_columns(
        plate_speed=(pl.col("vxf")**2+pl.col("vyf")**2+pl.col("vzf")**2)**0.5*0.681818
    )
    
    df=df.with_columns(
        max_launch_speed = (0.2023009*pl.col("plate_speed")+(1+0.2023009)*pl.col("bat_speed")),
        squared_up_percent=pl.col("launch_speed")/(0.2023009*pl.col("plate_speed")+(1+0.2023009)*pl.col("bat_speed"))
    )

    return df

def lin_reg_by_hand(fit_x,fit_y):
    tot_fit = np.concatenate([fit_x,fit_y],axis=1)
    n_x = fit_x.shape[1]
   
    fit_mu = np.mean(tot_fit,axis=0)
    fit_sig = np.cov(tot_fit,rowvar=False)
   
    x_mu = fit_mu[:n_x]
    x_sig = fit_sig[:n_x, :n_x]
   
    y_mu = fit_mu[n_x:]
    y_sig = fit_sig[n_x:, n_x:]
   
    cross_cov_y_x = fit_sig[n_x:, :n_x]
    cross_cov_x_y = cross_cov_y_x.transpose()

    beta = cross_cov_y_x @ np.linalg.inv(x_sig)

    return fit_mu,fit_sig,x_mu,x_sig,y_mu,y_sig,cross_cov_y_x,cross_cov_x_y,beta

def cov_to_corr(cov_matrix):
    std_devs = np.sqrt(np.diag(cov_matrix))    
    return cov_matrix[0,1] / (std_devs[0] * std_devs[1])

def entropy_by_hand(p):
    import math
    import numpy as np
    p_safe = np.where(p > 0, p, 1e-10)
    result = np.vectorize(math.log2)(p_safe)
    return (-p_safe * result).sum()

def evaluate_bivariate_normal_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho, norm_it):
    inputs = [x, y, mu_x, mu_y, sigma_x, sigma_y, rho]
    inputs = [np.asarray(i) for i in inputs]

    valid_mask = np.all([np.isfinite(i) for i in inputs], axis=0)
   
    pdf = np.full_like(x, np.nan, dtype=float)
   
    x_valid, y_valid = x[valid_mask], y[valid_mask]
    mu_x_valid, mu_y_valid = mu_x[valid_mask], mu_y[valid_mask]
    sigma_x_valid, sigma_y_valid = sigma_x[valid_mask], sigma_y[valid_mask]
    rho_valid = rho[valid_mask]
   
    coeff = 1 / (2 * np.pi * sigma_x_valid * sigma_y_valid * np.sqrt(1 - rho_valid**2))
   
    z_x = (x_valid - mu_x_valid) / sigma_x_valid
    z_y = (y_valid - mu_y_valid) / sigma_y_valid
   
    exponent = -(z_x**2 - 2*rho_valid*z_x*z_y + z_y**2) / (2 * (1 - rho_valid**2))
   
    pdf[valid_mask] = coeff * np.exp(exponent)

    if norm_it:
        z_x_max = np.zeros_like(mu_x_valid)
        z_y_max = np.zeros_like(mu_y_valid)
        exponent_max = -(z_x_max**2 - 2*rho_valid*z_x_max*z_y_max + z_y_max**2) / (2 * (1 - rho_valid**2))
        max_pdf = coeff * np.exp(exponent_max)
        pdf[valid_mask] /= max_pdf
   
    return pdf,valid_mask

def additional_columns_one(df):
    """from my kaggle submission"""
    # simple data cleaning on arm angle
    df["arm_angle_observed"] = df["arm_angle"].replace("--",np.nan).astype(float).values

    # use expected arm angle for missing data points
    x_vars = ["release_pos_x","release_pos_z","release_extension"]
    y_vars = ["arm_angle_observed"]
   
    fit_mu,fit_sig,x_mu,x_sig,y_mu,y_sig,cross_cov_y_x,cross_cov_x_y,beta = lin_reg_by_hand(
        fit_x = df.dropna(subset=x_vars+y_vars)[x_vars].to_numpy(),
        fit_y = df.dropna(subset=x_vars+y_vars)[y_vars].to_numpy(),
    )
   
    df["arm_angle_expected"] = y_mu + (df[x_vars] - x_mu) @ beta.T
    df["arm_angle_final"] = df["arm_angle_observed"].fillna(df["arm_angle_expected"]).fillna(np.nanmean(df["arm_angle_observed"]))

    # flip sign based on batter and pitcher handedness
    for col in ["pfx_x","release_pos_x"]:
        df[f"{col}_pit_flip"] = np.where(df["p_throws"]=="R",df[col].mul(-1),df[col])
        df[f"{col}_bat_flip"] = np.where(df["stand"]=="L",df[col].mul(-1),df[col])

    # horz and vert as popularly understood (flipped for pitcher handedness and in units of inches)
    df["horz"] = df["pfx_x_pit_flip"].mul(12)
    df["vert"] = df["pfx_z"].mul(12)

    # determines movement angle and angle relative to arm angle
    df["movement_angle"] = ((np.degrees(np.arctan2(df["horz"],df["vert"])) + 90)%360).astype(float)
    df["movement_angle_delta"] = df["movement_angle"].values - df["arm_angle_final"].values

    # total movement
    df["total_break"] = (df["horz"].values**2 + df["vert"].values**2)**0.5

    return df

def rotate_pitch_trajectory(
    dataFull,
    ball_cols,
    batter_cols=["batter_head_x", "batter_head_y", "batter_head_z"],
    release_cols=["release_pos_x_mean", "y0_mean", "release_pos_z_mean"],
    focal_length=0,
):
    
    batter_pos = dataFull[batter_cols].astype(float).to_numpy()
    ball_pos = dataFull[ball_cols].astype(float).to_numpy()
    release_pos = dataFull[release_cols].astype(float).to_numpy()
    
    translated_coords = ball_pos - batter_pos
    translated_release = release_pos - batter_pos
    
    direction = translated_release
    
    rot_y = -(np.arctan2(direction[:, 0], direction[:, 1]))
    rot_x = np.arctan2(direction[:, 2], np.sqrt(direction[:, 0]**2 + direction[:, 1]**2))
    
    cos_y, sin_y = np.cos(rot_y), np.sin(rot_y)
    cos_x, sin_x = np.cos(rot_x), np.sin(rot_x)
    
    rotation_y = np.array([
        [cos_y, -sin_y, np.zeros_like(rot_y)],
        [sin_y, cos_y, np.zeros_like(rot_y)],
        [np.zeros_like(rot_y), np.zeros_like(rot_y), np.ones_like(rot_y)]
    ]).transpose(2, 0, 1)
    
    rotation_x = np.array([
        [np.ones_like(rot_x), np.zeros_like(rot_x), np.zeros_like(rot_x)],
        [np.zeros_like(rot_x), cos_x, -sin_x],
        [np.zeros_like(rot_x), sin_x, cos_x]
    ]).transpose(2, 0, 1)
    
    rotated_coords = np.einsum("ijk,ikl,il->ij", rotation_x, rotation_y, translated_coords)
    
    return rotated_coords[:, 0], rotated_coords[:, 2]

def add_trajectory_info(dataFull):
    
    dataFull = add_release_metrics(dataFull)
    
    dataFull["batter_head_x"] = np.where(
        dataFull["stand"]=="L",
        2.5,
        -2.5
    )
    
    dataFull["batter_head_y"] = -1.0
    dataFull["batter_head_z"] = 4.75
    
    for release_col in ["release_pos_x","y0","release_pos_z"]:
        dataFull[f"{release_col}_mean"] = (
            dataFull[["pitcher","game_year","stand",release_col]]
            .groupby(["pitcher","game_year","stand"])
            .transform("mean")
        )
        
    tdec_list = [int(_tdec) for _tdec in np.linspace(10,350,35).tolist()]

    tdec_keeps=tdec_list
    
    dataFull["arc_distance_10_to_250"] = np.zeros(len(dataFull))
    
    for i, tdec_ms in enumerate(tdec_list):
        tdec_ms=int(tdec_ms)
        tdec_val = tdec_ms/1000

        tdec_fwd = tdec_val
        dataFull[f"x_{tdec_ms}"] = dataFull["release_pos_x"] + (0.5 * (dataFull["vxR"] + (dataFull["vxR"] + dataFull["ax"]*tdec_fwd))) * tdec_fwd
        dataFull[f"y_{tdec_ms}"] = dataFull["yR"] + (0.5 * (dataFull["vyR"] + (dataFull["vyR"] + dataFull["ay"]*tdec_fwd))) * tdec_fwd
        dataFull[f"z_{tdec_ms}"] = dataFull["release_pos_z"] + (0.5 * (dataFull["vzR"] + (dataFull["vzR"] + dataFull["az"]*tdec_fwd))) * tdec_fwd

        dataFull[f"visual_angle_{tdec_ms}"] = 2*np.arctan2(2.9/12,2*dataFull[f"y_{tdec_ms}"])*180/np.pi
        
        dir=""
        x_new, y_new = rotate_pitch_trajectory(
            dataFull,
            [f"x_{tdec_ms}{dir}", f"y_{tdec_ms}{dir}", f"z_{tdec_ms}{dir}"]
        )
        
        dataFull[f"x_{tdec_ms}_rotated"] = x_new
        dataFull[f"z_{tdec_ms}_rotated"] = y_new

        if tdec_ms>tdec_keeps[0]:
            dataFull[f"x_distance_10_to_{tdec_ms}"] = dataFull[f"x_{tdec_ms}_rotated"] - dataFull["x_10_rotated"]
            dataFull[f"z_distance_10_to_{tdec_ms}"] = dataFull[f"z_{tdec_ms}_rotated"] - dataFull["z_10_rotated"]
            dataFull[f"tot_distance_10_to_{tdec_ms}"] = (
                dataFull[f"x_distance_10_to_{tdec_ms}"]**2+dataFull[f"z_distance_10_to_{tdec_ms}"]**2
            )**0.5

        if tdec_ms > 10 and tdec_ms <= 250:
            dataFull["arc_distance_10_to_250"] += np.sqrt(
                (dataFull[f"x_{tdec_list[i]}_rotated"] - dataFull[f"x_{tdec_list[i-1]}_rotated"])**2 + 
                (dataFull[f"z_{tdec_list[i]}_rotated"] - dataFull[f"z_{tdec_list[i-1]}_rotated"])**2
            )
            
    dataFull["arc_depth_10_to_250"] = dataFull["arc_distance_10_to_250"] / dataFull["tot_distance_10_to_250"]
    
    return dataFull

def add_usage_rates(dataFull):

    for col in ["balls","strikes"]:
        dataFull[col]=dataFull[col].astype(int)
    dataFull["count_type"] = (
        np.where(
            (dataFull["strikes"]>=2)&(dataFull["balls"]>=3),
            "full",
            np.where(
                (dataFull["balls"]==dataFull["strikes"]),
                "even",
                np.where(
                    (dataFull["strikes"]>dataFull["balls"]),
                    "ahead",
                    np.where(
                        (dataFull["balls"]>dataFull["strikes"]),
                        "behind",
                        "null"
                    )
                )
            )
        )
    )

    def usage_portion(dataFull):
        dataFull["total_thrown_by_count"] = (
            dataFull[["release_speed","pitcher","game_year","count_type","stand"]]
            .groupby(["pitcher","game_year","count_type","stand"])
            .transform("count")
        )
        
        dataFull["thrown_by_count"] = (
            dataFull[["release_speed","pitcher","pitch_type","game_year","count_type","stand"]]
            .groupby(["pitcher","pitch_type","game_year","count_type","stand"])
            .transform("count")
        )
        
        dataFull["usage_by_count"] = dataFull["thrown_by_count"] / dataFull["total_thrown_by_count"]
    
        metrics=["usage_by_count"]
        group_cols = ["pitcher", "pitch_type", "game_year", "stand","count_type"]
        group_cols_two = ["pitcher","game_year","stand","count_type"]
        pitch_keeps = list(dataFull["pitch_type"].unique())
    
        for metric_i in metrics:
            for pitch_type in pitch_keeps:
                dataFull[f"{metric_i}_mean_{pitch_type}_init_init"] = list(dataFull[[metric_i]+group_cols].groupby(group_cols).transform("mean")[metric_i].values)
                dataFull[f"{metric_i}_mean_{pitch_type}_init"] = np.where(dataFull["pitch_type"]==pitch_type,dataFull[f"{metric_i}_mean_{pitch_type}_init_init"],np.nan)
                dataFull[f"{metric_i}_mean_{pitch_type}"] = dataFull[[f"{metric_i}_mean_{pitch_type}_init"]+group_cols_two].groupby(group_cols_two).transform(np.nanmean)
    
        drop_cols = [col for col in dataFull.columns if col.endswith("_init")]
        return dataFull[[col for col in dataFull.columns if col not in drop_cols]]

    dataFull=usage_portion(dataFull)

    dataFull["mean_velo"] = (
        dataFull
        [["pitcher","pitch_type","game_year","stand","release_speed"]]
        .groupby(["pitcher","pitch_type","game_year","stand"])
        .transform("mean")
    )
    
    dataFull["typical_fb_usage"]=dataFull[["usage_by_count_mean_SI","usage_by_count_mean_FF"]].sum(axis="columns")
    
    dataFull.loc[
    ((dataFull["mean_velo"]>=92.5)&(dataFull["pitch_type"]=="FC")&(dataFull["typical_fb_usage"].add(0.1)<dataFull["usage_by_count_mean_FC"])),
    "pitch_group"]="fastball"
    dataFull.loc[
    ((dataFull["mean_velo"]>=92.5)&(dataFull["pitch_type"]=="FC")&(dataFull["typical_fb_usage"].add(0.1)<dataFull["usage_by_count_mean_FC"])),
    "pitch_type"]="HC"
    
    dataFull=usage_portion(dataFull)

    return dataFull

def add_release_metrics(df):

    df = (
        df
        .assign(
            y0 = 50,
            yR=lambda df: 60.5 - df["release_extension"],
            tR=lambda df: (-df["vy0"] - np.sqrt(df["vy0"]**2 - 2 * df["ay"] * (50 - df["yR"]))) / df["ay"],
            vxR=lambda df: df["vx0"] + df["ax"] * df["tR"],
            vyR=lambda df: df["vy0"] + df["ay"] * df["tR"],
            vzR=lambda df: df["vz0"] + df["az"] * df["tR"],
            tf=lambda df: (-df["vyR"] - np.sqrt(df["vyR"]**2 - 2 * df["ay"] * (df["yR"] - 17/12))) / df["ay"],
        )
        .replace([-np.inf,np.inf],np.nan)
        .reset_index(drop=True)
    )

    return(df)

def add_clusters(
    df,
    n_clusters=10,
    pred_vars = ["release_speed","horz","vert","movement_angle_delta"],
):
    from sklearn.mixture import GaussianMixture
    
    fit_x = df[pred_vars].dropna().to_numpy()
    
    pitch_cluster_model = GaussianMixture(n_components=n_clusters, random_state=3024)
    pitch_cluster_model.fit(fit_x)
    
    pred_data = df[pred_vars].dropna()
    pred_ind = pred_data.index
    modeled_types = pitch_cluster_model.predict(pred_data)
    modeled_probs = pitch_cluster_model.predict_proba(pred_data)
    
    df["pitch_cluster"] = None
    df.loc[pred_ind,"pitch_cluster"] = modeled_types
    df.loc[pred_ind,"pitch_cluster"] = df.loc[pred_ind,"pitch_cluster"].astype(int)
    
    for i, pred_type in enumerate(np.arange(modeled_probs.shape[1])):
        df[f"cluster_{i}_weight"] = np.nan
        df.loc[pred_ind,f"cluster_{i}_weight"] = modeled_probs[:,i]

    df = df.dropna(subset=["pitch_cluster"])
    df["pitch_cluster"]=df["pitch_cluster"].astype(int)

    return df

def refit_swing_models_full(df):

    df=df.with_columns(
        int_y=df["intercept_ball_minus_batter_pos_y_inches"],
        int_x=df["intercept_ball_minus_batter_pos_x_inches"],
        sz_diff=df["sz_top"]-df["sz_bot"],
        batter_pitch_group=df["batter"].cast(pl.String)+"_"+df["pitch_group"].cast(pl.String)
    )
    
    barrels = (
        df
        .filter(
            pl.col("estimated_woba_using_speedangle")>0.7
        )
        .filter(
            pl.col("launch_speed")>90
        )
        .drop_nulls(subset=[
            "swing_path_tilt",
            "int_y",
            "bat_speed",
            "plate_x",
            "plate_z",
            "tf",
            "batter",
            "sz_top",
            "sz_bot",
            "sz_diff",
            "pitch_group"
        ])
    )
    
    swing_df=(
        df
        .filter(pl.col("batter_pitch_group").is_in(barrels["batter_pitch_group"].unique().to_list()))
        # .filter(pl.col("batter").is_in(barrels["batter"].unique().to_list()))
        .drop_nulls(subset=[
            "swing_path_tilt",
            "int_y",
            "bat_speed",
            "plate_x",
            "plate_z",
            "tf",
            "batter",
            "sz_top",
            "sz_bot",
            "sz_diff",
            "pitch_group"
        ])
    )

    ### TILT MODEL ###

    tilt_model = bmb.Model(
        "swing_path_tilt ~ 1 + sz_top + sz_diff + (1+plate_z|batter/pitch_group)",
        data = barrels.to_pandas(),
        family = "gaussian",
    )
    
    tilt_idata = tilt_model.fit(
        500, tune=500,
        chains=4, cores=4,
        nuts_sampler="nutpie",
        random_seed=3024,
        idata_kwargs={"log_likelihood": True}    
    )

    tilt_pps_barrels = tilt_model.predict(
        tilt_idata,
        kind="pps",
        data=barrels.to_pandas(),
        inplace=False,
    )
    
    barrels=barrels.with_columns(
        swing_path_tilt_pred=tilt_pps_barrels.posterior_predictive.swing_path_tilt.mean(("chain","draw")).data
    )
    
    barrels=barrels.with_columns(
        swing_path_tilt_error=barrels["swing_path_tilt"]-barrels["swing_path_tilt_pred"]
    )
    
    barrels=barrels.with_columns(
        swing_path_tilt_error_abs=(barrels["swing_path_tilt"]-barrels["swing_path_tilt_pred"]).abs()
    )

    tilt_pps = tilt_model.predict(
        tilt_idata,
        kind="pps",
        data=swing_df.to_pandas(),
        inplace=False,
    )
    
    swing_df=swing_df.with_columns(
        swing_path_tilt_pred=tilt_pps.posterior_predictive.swing_path_tilt.mean(("chain","draw")).data
    )
    
    swing_df=swing_df.with_columns(
        swing_path_tilt_error=swing_df["swing_path_tilt"]-swing_df["swing_path_tilt_pred"]
    )
    
    swing_df=swing_df.with_columns(
        swing_path_tilt_error_abs=(swing_df["swing_path_tilt"]-swing_df["swing_path_tilt_pred"]).abs()
    )

    ### Y INTERCEPT MODEL ###

    intercept_model = bmb.Model(
        "int_y ~ 1 + (1+tf+plate_x_bat_flip|batter/pitch_group)",
        data=barrels.to_pandas(),
        family = "gaussian",
    )
    
    intercept_idata = intercept_model.fit(
        500, tune=500,
        chains=4, cores=4,
        nuts_sampler="nutpie",
        random_seed=3024,
        idata_kwargs={"log_likelihood": True}    
    )

    intercept_pps_barrels = intercept_model.predict(
        intercept_idata,
        kind="pps",
        data=barrels.to_pandas(),
        inplace=False,
    )
    
    barrels=barrels.with_columns(
        int_y_pred=intercept_pps_barrels.posterior_predictive.int_y.mean(("chain","draw")).data
    )
    
    barrels=barrels.with_columns(
        int_y_error=barrels["int_y"]-barrels["int_y_pred"]
    )
    
    barrels=barrels.with_columns(
        int_y_error_abs=(barrels["int_y"]-barrels["int_y_pred"]).abs()
    )

    intercept_pps_barrels = intercept_model.predict(
        intercept_idata,
        kind="pps",
        data=swing_df.to_pandas(),
        inplace=False,
    )
    
    swing_df=swing_df.with_columns(
        int_y_pred=intercept_pps_barrels.posterior_predictive.int_y.mean(("chain","draw")).data
    )
    
    swing_df=swing_df.with_columns(
        int_y_error=swing_df["int_y"]-swing_df["int_y_pred"]
    )
    
    swing_df=swing_df.with_columns(
        int_y_error_abs=(swing_df["int_y"]-swing_df["int_y_pred"]).abs()
    )

    ### BAT SPEED MODEL
    
    bat_speed_model = bmb.Model(
        "bat_speed ~ 1 + (1+int_y|batter)",
        data=barrels.to_pandas(),
        family = "gaussian",
    )
    
    bat_speed_idata = bat_speed_model.fit(
        500, tune=500,
        chains=4, cores=4,
        nuts_sampler="nutpie",
        random_seed=3024,
        idata_kwargs={"log_likelihood": True}    
    )

    bat_speed_pps_barrels = bat_speed_model.predict(
        bat_speed_idata,
        kind="pps",
        data=barrels.to_pandas(),
        inplace=False,
    )
    
    barrels=barrels.with_columns(
        bat_speed_pred=bat_speed_pps_barrels.posterior_predictive.bat_speed.mean(("chain","draw")).data
    )
    
    barrels=barrels.with_columns(
        bat_speed_error=barrels["bat_speed"]-barrels["bat_speed_pred"]
    )
    
    barrels=barrels.with_columns(
        bat_speed_error_abs=(barrels["bat_speed"]-barrels["bat_speed_pred"]).abs()
    )

    bat_speed_pps_swings = bat_speed_model.predict(
        bat_speed_idata,
        kind="pps",
        data=swing_df.to_pandas(),
        inplace=False,
    )
    
    swing_df=swing_df.with_columns(
        bat_speed_pred=bat_speed_pps_swings.posterior_predictive.bat_speed.mean(("chain","draw")).data
    )
    
    swing_df=swing_df.with_columns(
        bat_speed_error=swing_df["bat_speed"]-swing_df["bat_speed_pred"]
    )
    
    swing_df=swing_df.with_columns(
        bat_speed_error_abs=(swing_df["bat_speed"]-swing_df["bat_speed_pred"]).abs()
    )

    swing_df.write_ipc("swing_df_2024.feather")

    return barrels, swing_df

def add_cluster_usage_rates(dataFull,pitch_type_column="pitch_cluster"):

    for col in ["balls","strikes"]:
        dataFull[col]=dataFull[col].astype(int)
    dataFull["count_type"] = (
        np.where(
            (dataFull["strikes"]>=2)&(dataFull["balls"]>=3),
            "full",
            np.where(
                (dataFull["balls"]==dataFull["strikes"]),
                "even",
                np.where(
                    (dataFull["strikes"]>dataFull["balls"]),
                    "ahead",
                    np.where(
                        (dataFull["balls"]>dataFull["strikes"]),
                        "behind",
                        "null"
                    )
                )
            )
        )
    )

    def usage_portion(dataFull,pitch_type_column):
        dataFull["total_thrown_by_count"] = (
            dataFull[["release_speed","pitcher","game_year","count_type","stand"]]
            .groupby(["pitcher","game_year","count_type","stand"])
            .transform("count")
        )
        
        dataFull["thrown_by_count"] = (
            dataFull[["release_speed","pitcher",pitch_type_column,"game_year","count_type","stand"]]
            .groupby(["pitcher",pitch_type_column,"game_year","count_type","stand"])
            .transform("count")
        )
        
        dataFull["usage_by_count"] = dataFull["thrown_by_count"] / dataFull["total_thrown_by_count"]
    
        metrics=["usage_by_count"]
        group_cols = ["pitcher", pitch_type_column, "game_year", "stand","count_type"]
        group_cols_two = ["pitcher","game_year","stand","count_type"]
        pitch_keeps = list(dataFull[pitch_type_column].unique())
    
        for metric_i in metrics:
            for pitch_cluster in pitch_keeps:
                dataFull[f"{metric_i}_mean_{pitch_cluster}_init_init"] = list(dataFull[[metric_i]+group_cols].groupby(group_cols).transform("mean")[metric_i].values)
                dataFull[f"{metric_i}_mean_{pitch_cluster}_init"] = np.where(dataFull[pitch_type_column]==pitch_cluster,dataFull[f"{metric_i}_mean_{pitch_cluster}_init_init"],np.nan)
                dataFull[f"{metric_i}_mean_{pitch_cluster}"] = dataFull[[f"{metric_i}_mean_{pitch_cluster}_init"]+group_cols_two].groupby(group_cols_two).transform(np.nanmean)
    
        drop_cols = [col for col in dataFull.columns if col.endswith("_init")]
        return dataFull[[col for col in dataFull.columns if col not in drop_cols]]

    dataFull=usage_portion(dataFull,pitch_type_column)

    return dataFull

def bat_flip_metric(dataFull,flip_met):
    return np.where(dataFull["stand"]=="L",dataFull[flip_met].mul(-1),dataFull[flip_met])
    
def release_angles_and_zone_angles(dataFull):
    for flip_met in [
        "x_10",
        "x_distance_10_to_250",
        "x_distance_10_to_300",
        "x_10_rotated",
        "x_50_rotated",
        "x_100_rotated",
        "x_150_rotated",
        "x_200_rotated",
        "x_250_rotated",
        "x_300_rotated",
        "x_350_rotated",
    ]:
        dataFull[f"{flip_met}_bat_flip"] = bat_flip_metric(dataFull,flip_met)
        
    dataFull["sz_center_z"] = 2.5
    dataFull["sz_center_x"] = 0
    
    dataFull["angle_to_zone_init"] = np.arctan2(
        dataFull[f"x_10_rotated_bat_flip"].abs()-dataFull["sz_center_x"],
        dataFull[f"z_10_rotated"]-dataFull["sz_center_z"]
    )
    
    dataFull["release_angle_visual_init"] = np.arctan2(
        dataFull[f"x_10_rotated_bat_flip"].abs()-dataFull[f"x_50_rotated_bat_flip"],
        dataFull[f"z_10_rotated"]-dataFull[f"z_50_rotated"]
    )
    
    relative_point = "10"
    
    dataFull["angle_to_zone"] = np.arctan2(
        dataFull[f"x_{relative_point}_rotated_bat_flip"].abs()-dataFull["sz_center_x"],
        dataFull[f"z_{relative_point}_rotated"]-dataFull["sz_center_z"]
    )
    
    dataFull["distance_to_zone"] = (
        (dataFull[f"x_10_rotated_bat_flip"]-dataFull["sz_center_x"])**2+
        (dataFull[f"z_10_rotated"]-dataFull["sz_center_z"])**2
    )**0.5

    return dataFull

def visual_angle_and_relative_distance(
    dataFull,
    relative_point = "10",
    final_point = "250"
):
    dataFull = release_angles_and_zone_angles(dataFull)
    
    dataFull["release_angle_visual"] = np.arctan2(
        dataFull[f"x_{relative_point}_rotated_bat_flip"].abs()-dataFull[f"x_{final_point}_rotated_bat_flip"],
        dataFull[f"z_{relative_point}_rotated"]-dataFull[f"z_{final_point}_rotated"]
    )
    
    dataFull["distance_from_hand"] = dataFull[f"tot_distance_{relative_point}_to_{final_point}"]
    
    dataFull["visual_angle_change"] = dataFull[f"visual_angle_{final_point}"] - dataFull[f"visual_angle_{relative_point}"]
    
    dataFull["distance_diff"] = dataFull[f"tot_distance_10_to_{final_point}"]/dataFull["distance_to_zone"]
    dataFull["release_angle_diff"] = dataFull["release_angle_visual"]-dataFull["angle_to_zone"]

    return dataFull

def add_predicted_pitch_types(
    df,
    x_vars=[
        "arm_angle_final",
        "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",
    ],
    pitch_type_column="pitch_cluster",
):
    """
    Just like DDZ but using the clusters from above rather than observed
    pitch types. So E[cluster|arm_angle,extension].
    
    """
    
    from sklearn.linear_model import LogisticRegression

    y_vars = [pitch_type_column]
    
    fit_x = df.dropna(subset=x_vars+y_vars)[x_vars]
    fit_y = df.dropna(subset=x_vars+y_vars)[y_vars]
    
    exp_cluster_model = LogisticRegression(multi_class="multinomial",random_state=3024)
    exp_cluster_model.fit(fit_x, fit_y)
    
    pred_data = df[x_vars].dropna()
    pred_ind = pred_data.index
    
    pred_types = exp_cluster_model.predict(pred_data)
    pred_probs = exp_cluster_model.predict_proba(pred_data)
    
    # determine predicted pitch cluster and pitch cluster probabilities
    df["pred_pitch_cluster"] = None
    df.loc[pred_ind,"pred_pitch_cluster"] = pred_types
    df.loc[pred_ind,"pred_pitch_cluster"] = df.loc[pred_ind,"pred_pitch_cluster"].astype(int)
    
    for i, pred_type in enumerate(exp_cluster_model.classes_):
        df[f"cluster_{i}_pred"] = None
        df.loc[pred_ind,f"cluster_{i}_pred"] = pred_probs[:,i]
        df[f"weighted_cluster_{i}_probability"] = df[f"cluster_{i}_pred"]*df[f"cluster_{int(i)}_weight"]
    
    df=df.dropna(subset=[col for col in df.columns if "weighted_cluster_" in col])
    
    # determine how surprising the given pitch cluster was and the entropy of the pitch cluster probability vector
    df["weighted_cluster_probability"] = df[[col for col in df.columns if col.startswith("weighted_cluster_") and col.endswith("_probability")]].sum(axis="columns")
    df["cluster_entropy"] = entropy_by_hand(df[[col for col in df.columns if col.startswith("cluster_") and col.endswith("_pred")]].to_numpy())
    
    df["cluster_probability"] = np.select(
        [df[pitch_type_column] == pt for pt in df[pitch_type_column].unique()],
        [df[f"cluster_{i}_pred"].fillna(0) for i in df[pitch_type_column].astype(int).unique()],
        default=None)

    return df

def add_predicted_pitch_types_catboost(
    df,
    x_vars=[
        "arm_angle_final",
        "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",
    ],
    refit=True,
    catboost_verbose=True,
    pitch_type_column="pitch_cluster",
    predicted_pitch_types_model_name="pitch_cluster_catboost",
):
    import catboost as cb
    
    y_vars = [pitch_type_column]
    
    exp_cluster_model = cb.CatBoostClassifier(
        loss_function="MultiClass",
        verbose = catboost_verbose,
        random_seed = 3024,
        cat_features=[col for col in x_vars if col in ["p_throws","stand","balls","strikes"]],
    )

    if refit:
        fit_x = df.dropna(subset=x_vars+y_vars)[x_vars]
        fit_y = df.dropna(subset=x_vars+y_vars)[y_vars]
        exp_cluster_model.fit(fit_x, fit_y)
        exp_cluster_model.save_model(f"{predicted_pitch_types_model_name}.cbm",format="cbm")
    else:
        exp_cluster_model.load_model(f"{predicted_pitch_types_model_name}.cbm",format="cbm")
    
    pred_data = df[x_vars].dropna()
    pred_ind = pred_data.index
    
    pred_types = exp_cluster_model.predict(pred_data)
    pred_probs = exp_cluster_model.predict_proba(pred_data)
    
    # determine predicted pitch cluster and pitch cluster probabilities
    df["pred_pitch_cluster_cb"] = None
    df.loc[pred_ind,"pred_pitch_cluster_cb"] = pred_types
    df.loc[pred_ind,"pred_pitch_cluster_cb"] = df.loc[pred_ind,"pred_pitch_cluster_cb"].astype(int)
    
    for i, pred_type in enumerate(exp_cluster_model.classes_):
        df[f"cluster_{i}_pred_cb"] = None
        df.loc[pred_ind,f"cluster_{i}_pred_cb"] = pred_probs[:,i]
    
    df["cluster_probability_cb"] = np.select(
        [df[pitch_type_column] == pt for pt in df[pitch_type_column].unique()],
        [df[f"cluster_{i}_pred_cb"].fillna(0) for i in df[pitch_type_column].astype(int).unique()],
        default=None)

    return df

def add_movement_vs_expected(
    df,
    x_vars=[
        "arm_angle_final",
        "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",        
    ],
    y_vars = ["horz","vert"],
    pitch_type_column="pitch_cluster",
):    
    for i, cluster in enumerate(df[pitch_type_column].dropna().unique()):
        clust_ = df.loc[(df[pitch_type_column]==cluster)]
       
        fit_mu,fit_sig,x_mu,x_sig,y_mu,y_sig,cross_cov_y_x,cross_cov_x_y,beta = lin_reg_by_hand(
            fit_x = clust_.dropna(subset=x_vars+y_vars)[x_vars].to_numpy(),
            fit_y = clust_.dropna(subset=x_vars+y_vars)[y_vars].to_numpy(),
        )
       
        pred_mu = y_mu + (df[x_vars] - x_mu) @ beta.T
        pred_sig = y_sig - beta @ cross_cov_x_y
    
        for i,y_var in enumerate(y_vars):
            df[f"cluster_{cluster}_pred_mu_{y_var}"] = pred_mu.to_numpy()[:,i]
            df[f"cluster_{cluster}_pred_var_{y_var}"] = pred_sig[i,i]
        if len(y_vars)==2:
            df[f"cluster_{cluster}_pred_corr_{y_vars[0]}_{y_vars[1]}"] = cov_to_corr(pred_sig)

    return df

def labeled_swing_df(
    swing_df,
    plane_lower_quantile_threshold=0.5,
    plane_upper_quantile_threshold=0.5,
    time_lower_quantile_threshold=0.5,
    time_upper_quantile_threshold=0.5,
):
    swing_df=swing_df.with_columns(
        plane_lower_bound=pl.col("swing_path_tilt_error").quantile(plane_lower_quantile_threshold).over("pitch_group"),
        plane_upper_bound=pl.col("swing_path_tilt_error").quantile(plane_upper_quantile_threshold).over("pitch_group"),
        time_lower_bound=pl.col("int_y_error").quantile(time_lower_quantile_threshold).over("pitch_group"),
        time_upper_bound=pl.col("int_y_error").quantile(time_upper_quantile_threshold).over("pitch_group")
    )
    
    swing_df=swing_df.with_columns(
        on_plane=(swing_df["swing_path_tilt_error"].is_between(
            lower_bound=swing_df["plane_lower_bound"],
            upper_bound=swing_df["plane_upper_bound"]
        )).cast(pl.Int32),
        over=(swing_df["swing_path_tilt_error"]<=pl.col("plane_lower_bound")).cast(pl.Int32),
        under=(swing_df["swing_path_tilt_error"]>=pl.col("plane_upper_bound")).cast(pl.Int32),
    
        on_time=(swing_df["int_y_error"].is_between(
            lower_bound=swing_df["time_lower_bound"],
            upper_bound=swing_df["time_upper_bound"]
        )).cast(pl.Int32),
        behind=(swing_df["int_y_error"]<=pl.col("time_lower_bound")).cast(pl.Int32),
        ahead=(swing_df["int_y_error"]>=pl.col("time_upper_bound")).cast(pl.Int32),
    )
    
    swing_df=swing_df.with_columns(
        plane_result=np.where(swing_df["under"]==1,"under",np.where(swing_df["over"]==1,"over","on_plane")),
        time_result=np.where(swing_df["ahead"]==1,"ahead",np.where(swing_df["behind"]==1,"behind","on_time")),
    )
    
    _plane_result = np.where(
        swing_df["pitch_group"].is_in(["fastball"]),
        np.where(
            swing_df["plane_result"]=="over",
            "on_plane",
            swing_df["plane_result"]
        ),
        swing_df["plane_result"]
    )
    
    _time_result = np.where(
        swing_df["pitch_group"].is_in(["fastball"]),
        np.where(
            swing_df["time_result"]=="ahead",
            "on_time",
            swing_df["time_result"]
        ),
        swing_df["time_result"]
    )
    
    swing_df=swing_df.with_columns(
        plane_result=_plane_result,
        time_result=_time_result,
    )
    
    _time_result=np.where(
        swing_df["bat_speed"]<0.8*swing_df["bat_speed_pred"],
        "ahead",
        swing_df["time_result"]
    )
    
    swing_df=swing_df.with_columns(
        time_result=_time_result,
    )
    
    swing_df=swing_df.with_columns(
        total_swing_result=pl.col("plane_result")+" - "+pl.col("time_result")
    )

    swing_df=swing_df.drop(["ahead","behind","over","under"]).with_columns(
        over=(swing_df["plane_result"]=="over").cast(pl.Int32),
        under=(swing_df["plane_result"]=="under").cast(pl.Int32),
        ahead=(swing_df["time_result"]=="ahead").cast(pl.Int32),
        behind=(swing_df["time_result"]=="behind").cast(pl.Int32),
    )

    return swing_df

def update_predicted_pitch_types(
    swing_df,
    pred_pitch_type_x_vars=[
        "arm_angle_final",
        "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",        
    ],
    refit=True,
    characteristic_x_vars=[
        "arm_angle_final",
        "release_extension",
    ],
    update_expected_movement=False,
    catboost_verbose=True,
    pitch_type_column="pitch_cluster",
    predicted_pitch_types_model_name="pitch_cluster_catboost",
):
    from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score, make_scorer, mean_squared_error, r2_score
    from scipy.stats import pearsonr
    
    swing_df=pl.from_pandas(add_predicted_pitch_types_catboost(
        swing_df.to_pandas(),
        pred_pitch_type_x_vars,
        refit,
        catboost_verbose,
        pitch_type_column=pitch_type_column,
        predicted_pitch_types_model_name=predicted_pitch_types_model_name,
    ))

    if update_expected_movement:
        swing_df = pl.from_pandas(add_movement_vs_expected(
            swing_df.to_pandas(),
            characteristic_x_vars,
            ["api_break_z_with_gravity","api_break_x_batter_in"],
            pitch_type_column=pitch_type_column,
        ))
    
        for cluster in swing_df[pitch_type_column].unique():
            swing_df = swing_df.with_columns(
                (pl.col("api_break_z_with_gravity")<pl.col(f"cluster_{cluster}_pred_mu_api_break_z_with_gravity"))
                    .cast(pl.Int32)
                .alias(f"cluster_{cluster}_under"),
                (pl.col("api_break_z_with_gravity")>pl.col(f"cluster_{cluster}_pred_mu_api_break_z_with_gravity"))
                    .cast(pl.Int32)
                .alias(f"cluster_{cluster}_over"),
            )

        swing_df = pl.from_pandas(add_movement_vs_expected(
            swing_df.to_pandas(),
            characteristic_x_vars,
            ["release_speed"],
            pitch_type_column=pitch_type_column
        ))
        
        for cluster in swing_df[pitch_type_column].unique():
            swing_df = swing_df.with_columns(
                (pl.col("release_speed")<pl.col(f"cluster_{cluster}_pred_mu_release_speed"))
                    .cast(pl.Int32)
                .alias(f"cluster_{cluster}_ahead"),
                (pl.col("release_speed")>pl.col(f"cluster_{cluster}_pred_mu_release_speed"))
                    .cast(pl.Int32)
                .alias(f"cluster_{cluster}_behind"),
            )
    
    pred_col = "_cb"

    for i,cluster in enumerate(swing_df[pitch_type_column].unique().to_list()):
        if i==0:
            under_prob=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_under"])
            over_prob=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_over"])
    
            behind_prob=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_behind"])
            ahead_prob=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_ahead"])
        else:
            under_prob+=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_under"])
            over_prob+=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_over"])
    
            behind_prob+=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_behind"])
            ahead_prob+=(swing_df[f"cluster_{cluster}_pred{pred_col}"]*swing_df[f"cluster_{cluster}_ahead"])
    
    swing_df = swing_df.with_columns(
        under_probability=np.minimum(1,under_prob),
        over_probability=np.minimum(1,over_prob),
        behind_probability=np.minimum(1,behind_prob),
        ahead_probability=np.minimum(1,ahead_prob),
    )

    under_auc = roc_auc_score(
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["under"].to_numpy(),
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["under_probability"].to_numpy()
    )
    
    ahead_auc = roc_auc_score(
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["ahead"].to_numpy(),
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["ahead_probability"].to_numpy()
    )
    
    total_auc = float(np.mean([under_auc,ahead_auc]))

    from sklearn.metrics import brier_score_loss
    under_brier = brier_score_loss(
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["under"].to_numpy(),
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["under_probability"].to_numpy()
    )
    
    ahead_brier = brier_score_loss(
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["ahead"].to_numpy(),
        swing_df.filter(~(pl.col("pitch_group")=="fastball"))["ahead_probability"].to_numpy()
    )

    total_brier = float(np.mean([under_brier,ahead_brier]))    
    
    return swing_df, under_auc, ahead_auc, total_auc, under_brier, ahead_brier, total_brier

def provide_optuna_results(swing_miss_model_opt):
    import optuna
    
    print("\n" + "="*70)
    print(f"OPTIMIZATION COMPLETED - BEST RESULTS")
    print("="*70)
    print(f"Best trial: {swing_miss_model_opt.best_trial.number}")
    print(f"Best AUC: {swing_miss_model_opt.best_trial.value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in swing_miss_model_opt.best_params.items():
        print(f"    {key}: {value}")
    
    try:
        fig = optuna.visualization.plot_optimization_history(swing_miss_model_opt)
        fig.show()
    
        fig = optuna.visualization.plot_param_importances(swing_miss_model_opt)
        fig.show()
    
    except (ImportError, ModuleNotFoundError):
        print("\nVisualization skipped - requires plotly and other dependencies")

def full_df_modifications(df):
    df=flip_batter_cols(df)
    df=add_event_cols(df)
    df=add_flight_metrics_and_squared_up_pct(df)
    df=find_competitive_swings(df)
    df=add_pitch_group(df)

    ## Yes I know I should just update these older helper functions to polars, but for now I'm just going to convert to and from so I can reuse
    
    df=pl.from_pandas(additional_columns_one(df.to_pandas()))
    df=pl.from_pandas(add_trajectory_info(df.to_pandas()))
    df=pl.from_pandas(add_usage_rates(df.to_pandas()))
    df=pl.from_pandas(add_clusters(df.to_pandas()))
    df=pl.from_pandas(add_cluster_usage_rates(df.to_pandas()))

    df = pl.from_pandas(visual_angle_and_relative_distance(
        dataFull=df.to_pandas(),
        relative_point="10",
        final_point="250",
    ))
    
    return df

def prep_swing_df(
    swing_df,
    characteristic_x_vars=["arm_angle_final","release_extension"],
    pitch_type_column="pitch_cluster",
):
    for keeper in ["arm_angle_final","release_extension","tot_distance_10_to_250","arc_depth_10_to_250","visual_angle_change"]:
        if f"{keeper}_orig" not in swing_df.columns:
            swing_df = swing_df.with_columns(
                pl.col(keeper).mul(1).alias(f"{keeper}_orig"),
            )
    
    swing_df=swing_df.drop_nulls(subset=[
        "under",
        "arm_angle_final",
        "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",
    ])
    
    swing_df = pl.from_pandas(add_movement_vs_expected(
        swing_df.to_pandas(),
        characteristic_x_vars,
        ["api_break_z_with_gravity","api_break_x_batter_in"],
        pitch_type_column=pitch_type_column,
    ))
    
    for cluster in swing_df[pitch_type_column].unique():
        swing_df = swing_df.with_columns(
            (pl.col("api_break_z_with_gravity")<pl.col(f"cluster_{cluster}_pred_mu_api_break_z_with_gravity"))
                .cast(pl.Int32)
            .alias(f"cluster_{cluster}_under"),
            (pl.col("api_break_z_with_gravity")>pl.col(f"cluster_{cluster}_pred_mu_api_break_z_with_gravity"))
                .cast(pl.Int32)
            .alias(f"cluster_{cluster}_over"),
        )
    
    swing_df = pl.from_pandas(add_movement_vs_expected(
        swing_df.to_pandas(),
        characteristic_x_vars,
        ["release_speed"],
        pitch_type_column=pitch_type_column,
    ))
    
    for cluster in swing_df[pitch_type_column].unique():
        swing_df = swing_df.with_columns(
            (pl.col("release_speed")<pl.col(f"cluster_{cluster}_pred_mu_release_speed"))
                .cast(pl.Int32)
            .alias(f"cluster_{cluster}_ahead"),
            (pl.col("release_speed")>pl.col(f"cluster_{cluster}_pred_mu_release_speed"))
                .cast(pl.Int32)
            .alias(f"cluster_{cluster}_behind"),
        )
    
    return swing_df

def add_arc_meas(dataFull,arc_lim):
        
    tdec_list = [int(_tdec) for _tdec in np.linspace(10,350,35).tolist()]

    tdec_keeps=tdec_list
    
    dataFull[f"arc_distance_10_to_{arc_lim}"] = np.zeros(len(dataFull))
    
    for i, tdec_ms in enumerate(tdec_list):
        if tdec_ms>arc_lim:
            break
        else:
            tdec_ms=int(tdec_ms)
            tdec_val = tdec_ms/1000
    
            if tdec_ms > 10 and tdec_ms <= arc_lim:
                dataFull[f"arc_distance_10_to_{arc_lim}"] += np.sqrt(
                    (dataFull[f"x_{tdec_list[i]}_rotated"] - dataFull[f"x_{tdec_list[i-1]}_rotated"])**2 + 
                    (dataFull[f"z_{tdec_list[i]}_rotated"] - dataFull[f"z_{tdec_list[i-1]}_rotated"])**2
                )
                
        dataFull[f"arc_depth_10_to_{arc_lim}"] = dataFull[f"arc_distance_10_to_{arc_lim}"] / dataFull[f"tot_distance_10_to_{arc_lim}"]

    dataFull[f"visual_angle_change_10_to_{arc_lim}"] = dataFull[f"visual_angle_{arc_lim}"] - dataFull[f"visual_angle_10"]
    
    return dataFull

def fit_swings_catboost(
    df,
    x_vars=[
        "arm_angle_final",
        # "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",
        "release_angle_visual",
        "p_throws",
    ],
    catboost_verbose=True,
    y_column="swing_or_take",
    init_data_split_frac=0.5,
    n_folds_cv=3,
    model_save_name="swing_or_take_sabersem",
):

    import catboost as cb
    from sklearn.metrics import f1_score, accuracy_score, log_loss, roc_auc_score, make_scorer
    from sklearn.model_selection import train_test_split, StratifiedKFold

    model_data = (
        df
        .select(x_vars+[y_column])
        .to_pandas()
        .dropna()
        .sample(frac=init_data_split_frac,random_state=3024)
    )
    
    X = model_data[x_vars]
    y = model_data[y_column]

    params = {
        "loss_function": "Logloss",
        "verbose": catboost_verbose,
        "random_seed": 3024,
        "cat_features": [col for col in x_vars if col in ["balls","strikes","stand","p_throws"]],
    }

    n_folds = n_folds_cv
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=3024)
    cv_scores = []
    
    for train_idx, valid_idx in cv.split(X, y):
        X_cv_train, X_cv_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_cv_train, y_cv_valid = y.iloc[train_idx], y.iloc[valid_idx]
        
        try:
            model = cb.CatBoostClassifier(**params)
            
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=[(X_cv_valid, y_cv_valid)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            y_pred = model.predict_proba(X_cv_valid)
            
            if y_pred.shape[1] >= 2:
                score = roc_auc_score(y_cv_valid, y_pred[:, 1])
                cv_scores.append(score)
            else:
                return 0.0

            model.save_model(f"{model_save_name}.cbm",format="cbm")
                
        except Exception as e:
            print(f"Error in training: {e}")
            return 0.0
    
    return sum(cv_scores) / len(cv_scores) if cv_scores else 0.0

def predict_swings_catboost(
    df,
    x_vars=[
        "arm_angle_final",
        # "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",
        "release_angle_visual",
        "p_throws",
    ],
    model_save_name="swing_or_take_sabersem",
    pred_col_name="swing_probability_sabersem"
):

    import catboost as cb
    
    params = {
        "loss_function": "Logloss",
        "random_seed": 3024,
        "cat_features": [col for col in x_vars if col in ["balls","strikes","stand","p_throws"]],
    }
    
    model = cb.CatBoostClassifier(**params)
    model.load_model(f"{model_save_name}.cbm",format="cbm")

    y_pred = model.predict_proba(df.select(x_vars).to_pandas())

    df=df.with_columns(pred_col_name=y_pred[:,0]).rename({"pred_col_name":pred_col_name})

    return df

def finalize_swing_df_and_df(
    swing_df,
    pitch_type_column,
    df_ipc_read=".\statcast_pickles\\statcast_2024_raw.feather",
):
    
    swing_df = pl.from_pandas(visual_angle_and_relative_distance(
        dataFull=swing_df.to_pandas(),
        relative_point="10",
        final_point="250",
    ))
    
    PITCH_TYPE_INT, PITCH_TYPE_DECODE = pd.factorize(swing_df["pitch_type"])
    swing_df=swing_df.with_columns(pitch_type_int=PITCH_TYPE_INT)
    
    swing_df=prep_swing_df(
        swing_df,
        characteristic_x_vars=["arm_angle_final","release_extension"],
        pitch_type_column=pitch_type_column,
    )
    
    swing_df.with_columns(
        pl.col("release_angle_visual").mul(180/np.pi).alias("release_angle_visual")
    )
    
    swing_df = pl.from_pandas(add_cluster_usage_rates(swing_df.to_pandas(),pitch_type_column=pitch_type_column))
    
    swing_df=swing_df.with_columns(
        simple_pitch_group=np.where(swing_df["pitch_group"].is_in(["slider","curve"]),"breaking",swing_df["pitch_group"])
    )
    
    swing_df=swing_df.with_columns(
        well_hit=(
            swing_df["estimated_woba_using_speedangle"]>=swing_df["estimated_woba_using_speedangle"].quantile(0.75)
        ).cast(pl.Int32),
    )
    
    df = full_df_modifications(
        pl.read_ipc(df_ipc_read)
        .filter(~(pl.col("pitch_type").is_null()|pl.col("pitch_type").is_in(["EP","PO","SC"])))
    )
    
    df = df.drop([col for col in ["bat_speed_pred","swing_or_take_modified"] if col in df.columns]).join(
        swing_df[["pitch_number","at_bat_number","game_pk","bat_speed_pred"]],
        on=["pitch_number","at_bat_number","game_pk"],
        how="left"
    ).with_columns(pl.col("bat_speed_pred").fill_null(0).alias("bat_speed_pred"))
    
    swing_or_take_modified = np.where(
        (df["bat_speed_pred"]>0)&(df["bat_speed"]<=0.9125*df["bat_speed_pred"]),
        "take",
        df["swing_or_take"]
    )
    
    df=df.with_columns(
        swing_or_take_modified=swing_or_take_modified,
    )
    
    return swing_df, df

def read_data_and_initialize(
    refit_swing_models,
    rerun_optuna,
    load_string_init="",
    statcast_feather_name=".\statcast_pickles\\statcast_2024_raw.feather",
    plane_lower_quantile_threshold=0.45,
    plane_upper_quantile_threshold=0.55,
    time_lower_quantile_threshold=0.45,
    time_upper_quantile_threshold=0.55,
    swing_df_feather_name="swing_df_2024.feather",
    pitch_type_column="pitch_cluster",
):
    if rerun_optuna:
        from datetime import datetime
        load_string=datetime.today().strftime("%Y_%m_%d")
    else:
        load_string=load_string_init
    
    if refit_swing_models:
        df = full_df_modifications(
            pl.read_ipc(statcast_feather_name)
            .filter(~(pl.col("pitch_type").is_null()|pl.col("pitch_type").is_in(["EP","PO","SC"])))
        )
        
        barrels, swing_df = refit_swing_models_full(df)
        
        swing_df=labeled_swing_df(
            swing_df,
            plane_lower_quantile_threshold=plane_lower_quantile_threshold,
            plane_upper_quantile_threshold=plane_upper_quantile_threshold,
            time_lower_quantile_threshold=time_lower_quantile_threshold,
            time_upper_quantile_threshold=time_upper_quantile_threshold,
        )
        
        del df, barrels
    else:
        swing_df = labeled_swing_df(
            pl.read_ipc(swing_df_feather_name),
            plane_lower_quantile_threshold=plane_lower_quantile_threshold,
            plane_upper_quantile_threshold=plane_upper_quantile_threshold,
            time_lower_quantile_threshold=time_lower_quantile_threshold,
            time_upper_quantile_threshold=time_upper_quantile_threshold,
        )
    
    swing_df, df = finalize_swing_df_and_df(
        swing_df,
        pitch_type_column,
        df_ipc_read=statcast_feather_name,
    )
    
    return load_string,swing_df,df

def add_predicted_pitch_types_catboost_v2(
    df,
    x_vars=[
        "arm_angle_final",
        "release_extension",
        "tot_distance_10_to_250",
        "arc_depth_10_to_250",
        "visual_angle_change",
    ],
    refit=True,
    catboost_verbose=True,
    pitch_type_column="pitch_cluster",
    predicted_pitch_types_model_name="pitch_cluster_catboost",
    prediction_suffix="_cb",
):
    import catboost as cb
    
    y_vars = [pitch_type_column]
    
    exp_cluster_model = cb.CatBoostClassifier(
        loss_function="MultiClass",
        verbose = catboost_verbose,
        random_seed = 3024,
        cat_features=[col for col in x_vars if col in ["p_throws","stand","balls","strikes"]],
    )

    if refit:
        fit_x = df.dropna(subset=x_vars+y_vars)[x_vars]
        fit_y = df.dropna(subset=x_vars+y_vars)[y_vars]
        exp_cluster_model.fit(fit_x, fit_y)
        exp_cluster_model.save_model(f"{predicted_pitch_types_model_name}.cbm",format="cbm")
    else:
        exp_cluster_model.load_model(f"{predicted_pitch_types_model_name}.cbm",format="cbm")
    
    pred_data = df[x_vars].dropna()
    pred_ind = pred_data.index
    
    pred_types = exp_cluster_model.predict(pred_data)
    pred_probs = exp_cluster_model.predict_proba(pred_data)
    
    # determine predicted pitch cluster and pitch cluster probabilities
    df[f"pred_pitch_cluster{prediction_suffix}"] = None
    df.loc[pred_ind,f"pred_pitch_cluster{prediction_suffix}"] = pred_types
    df.loc[pred_ind,f"pred_pitch_cluster{prediction_suffix}"] = df.loc[pred_ind,f"pred_pitch_cluster{prediction_suffix}"].astype(int)
    
    for i, pred_type in enumerate(exp_cluster_model.classes_):
        df[f"cluster_{i}_pred{prediction_suffix}"] = None
        df.loc[pred_ind,f"cluster_{i}_pred{prediction_suffix}"] = pred_probs[:,i]
    
    df[f"cluster_probability{prediction_suffix}"] = np.select(
        [df[pitch_type_column] == pt for pt in df[pitch_type_column].unique()],
        [df[f"cluster_{i}_pred{prediction_suffix}"].fillna(0) for i in df[pitch_type_column].astype(int).unique()],
        default=None)

    return df

def add_pred_movement_total(
    df,
    prediction_suffix,
    pitch_type_column="pitch_cluster",
):
    df=df.to_pandas()
    for pred_meas in ["api_break_z_with_gravity","api_break_x"]:
        df[f"cluster_pred_mu_{pred_meas}{prediction_suffix}"] = np.select(
            [df[f"pred_pitch_cluster{prediction_suffix}"] == pt for pt in df[pitch_type_column].unique()],
            [df[f"cluster_{i}_pred_mu_{pred_meas}"].fillna(0) for i in df[pitch_type_column].astype(int).unique()],
            default=None)
    return pl.from_pandas(df)

def calculate_cs_prob_pred(
    df,
    cs_gam,
    model_suffix="_init_traj_w_uncertainty",
):
    cs_prob_temp=np.zeros(len(df))
    for i,cluster in enumerate(df["pitch_cluster"].unique()):
        df=df.with_columns(
            plate_x_temp=df["plate_x"]-df["api_break_x"]+df[f"cluster_{cluster}_pred_mu_api_break_x"],
            plate_z_temp=df["plate_z"]+df["api_break_z_with_gravity"]-df[f"cluster_{cluster}_pred_mu_api_break_z_with_gravity"]
        )
        _cs_prob_temp=cs_gam.predict(df[["plate_x_temp","plate_z_temp"]])
        cs_prob_temp+=(df[f"cluster_{cluster}_pred{model_suffix}"].to_numpy()*_cs_prob_temp)

        df=(
            df
            .with_columns(
                _cs_prob_temp=_cs_prob_temp,
            )
            .drop([col for col in df.columns if f"cs_prob_cluster_{cluster}{model_suffix}" in col])
            .rename({"_cs_prob_temp":f"cs_prob_cluster_{cluster}{model_suffix}"})
        )
    
    df=(
        df
        .with_columns(
            cs_prob_temp=cs_prob_temp,
        )
        .drop([col for col in df.columns if f"cs_prob_wtd{model_suffix}" in col])
        .rename({"cs_prob_temp":f"cs_prob_wtd{model_suffix}"})
    )
    return df
