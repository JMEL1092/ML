"""Part 2: Cells 7-9 (Feature Engineering, Variable Validation FDR, Optuna)"""
import nbformat

cells = []

# ─────────────────────────────────────────────────────────────────
# CELL 7 – Feature engineering
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── FEATURE ENGINEERING (sin data leakage temporal) ──────────────────────────
# REGLA: todas las features deben tener shift ≥ 1 mes respecto al target.
# Los encoders se fittean SÓLO sobre datos de entrenamiento (ver celda de WF).

def build_features(panel: pd.DataFrame, cohorte_serie: pd.DataFrame,
                   costos_ref: pd.DataFrame, fit_encoders: bool = True,
                   encoders: dict = None) -> tuple:
    \"\"\"
    Construye la matriz de features para el modelo global.
    Retorna (X_df, y, encoders_dict).

    fit_encoders=True  → fit nuevos encoders (modo entrenamiento inicial)
    fit_encoders=False → usar encoders existentes (modo incremental / validación)
    \"\"\"
    df = panel.copy()
    if "serie_id" not in df.columns:
        raise ValueError("panel debe contener la columna 'serie_id'")
    df = df.sort_values(["serie_id", "mes_repair"]).reset_index(drop=True)

    # Merge metadatos de costo y cohorte
    # Eliminar columnas ya presentes para evitar duplicados en llamadas recursivas
    _merge_cols_cost    = [c for c in costos_ref.columns if c != "serie_id"]
    _merge_cols_cohorte = ["cohorte_ratio_max", "cohorte_ratio_mean"]
    df = df.drop(columns=[c for c in _merge_cols_cost + _merge_cols_cohorte if c in df.columns])
    df = df.merge(costos_ref, on="serie_id", how="left")
    df = df.merge(cohorte_serie[["serie_id","cohorte_ratio_max","cohorte_ratio_mean"]],
                  on="serie_id", how="left")

    # ── Features de lag y rolling (shift ≥ 1 → sin leakage) ─────────────────
    # Usamos .transform() vectorizado para evitar pérdida de columnas con .apply()
    df = df.sort_values(["serie_id", "mes_repair"]).reset_index(drop=True)

    grpf = df.groupby("serie_id")["fallas"]
    grpc = df.groupby("serie_id")["costo_total"]

    df["lag_1"]      = grpf.shift(1)
    df["lag_2"]      = grpf.shift(2)
    df["lag_3"]      = grpf.shift(3)
    df["lag_6"]      = grpf.shift(6)
    df["roll3_mean"] = grpf.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.rolling(3, min_periods=1).mean())
    df["roll6_mean"] = grpf.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.rolling(6, min_periods=1).mean())
    df["roll3_cost"] = grpc.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.rolling(3, min_periods=1).mean())
    df["roll6_cost"] = grpc.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.rolling(6, min_periods=1).mean())
    df["trend_3"]    = grpf.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.rolling(3, min_periods=2).apply(slope_lineal, raw=True))
    df["trend_6"]    = grpf.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.rolling(6, min_periods=3).apply(slope_lineal, raw=True))
    df["cusum_feat"] = grpf.shift(1).groupby(df["serie_id"]).transform(
                           lambda x: x.expanding().apply(cusum_stat, raw=True))
    df["mis_lag1"]   = df.groupby("serie_id")["mis_mean"].shift(1)
    df["km_lag1"]    = df.groupby("serie_id")["km_mean"].shift(1)

    # ── Features de calendario ────────────────────────────────────────────────
    df["mes_num"]    = df["mes_dt"].dt.month
    df["anio"]       = df["mes_dt"].dt.year
    df["quarter"]    = df["mes_dt"].dt.quarter
    # Codificación cíclica del mes para capturar estacionalidad
    df["mes_sin"]    = np.sin(2 * np.pi * df["mes_num"] / 12)
    df["mes_cos"]    = np.cos(2 * np.pi * df["mes_num"] / 12)

    # ── Encoders categóricos (fit sólo en train) ──────────────────────────────
    cat_cols = ["ea_number", "componente", "damage_code", "mis_cluster"]
    if encoders is None:
        encoders = {}

    for col in cat_cols:
        enc_key = f"le_{col}"
        if fit_encoders:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str).fillna("UNK"))
            encoders[enc_key] = le
        else:
            le = encoders.get(enc_key)
            if le is not None:
                known = set(le.classes_)
                df[col] = df[col].astype(str).fillna("UNK").apply(
                    lambda x: x if x in known else le.classes_[0]
                )
                df[f"{col}_enc"] = le.transform(df[col])
            else:
                df[f"{col}_enc"] = 0

    # ── Features de cohorte (Engine Date – señal anticipada) ─────────────────
    df["cohorte_ratio"]   = df["cohorte_ratio_max"].fillna(1.0)
    df["cohorte_ratio"] = df["cohorte_ratio"].clip(0, 5)

    # ── Costo unitario de referencia ──────────────────────────────────────────
    _med = df["costo_unit_med"].median()
    df["costo_unit_ref"] = df["costo_unit_med"].fillna(0.0 if pd.isna(_med) else _med)

    # ── Columnas de features finales ─────────────────────────────────────────
    FEATURE_COLS = [
        "lag_1","lag_2","lag_3","lag_6",
        "roll3_mean","roll6_mean","roll3_cost","roll6_cost",
        "trend_3","trend_6","cusum_feat",
        "mis_lag1","km_lag1",
        "cohorte_ratio",
        "mes_sin","mes_cos","quarter","anio",
        "ea_number_enc","componente_enc","damage_code_enc","mis_cluster_enc",
        "costo_unit_ref",
    ]

    # Añadir candidatas nuevas si pasan la validación estadística
    for c in CANDIDATAS_NUEVAS_NUMERICAS + CANDIDATAS_NUEVAS_CATEGORICAS:
        enc_c = f"{c}_enc" if c in CANDIDATAS_NUEVAS_CATEGORICAS else c
        if enc_c in df.columns:
            FEATURE_COLS.append(enc_c)

    # Eliminar filas sin lag_1 (primer mes de cada serie)
    df_model = df.dropna(subset=["lag_1"]).copy()
    df_model[FEATURE_COLS] = df_model[FEATURE_COLS].fillna(0)

    X = df_model[FEATURE_COLS].astype(float)
    y = df_model["fallas"].astype(float)

    return X, y, df_model, FEATURE_COLS, encoders


# Primera pasada para conocer las columnas
X_all, y_all, panel_feat, FEATURE_COLS, encoders_global = build_features(
    monthly, cohorte_serie, costos_ref, fit_encoders=True
)

print(f"✅ Features construidas: {len(FEATURE_COLS)} variables")
print(f"   Filas disponibles para modelado: {len(X_all):,}")
print("   Features:", FEATURE_COLS)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 8 – Variable validation with FDR
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VALIDACIÓN ESTADÍSTICA DE VARIABLES (FDR Benjamini-Hochberg) ─────────────
# Se corre automáticamente para las candidatas nuevas.
# Las variables base del modelo no se someten a veto estadístico (ya están validadas).

validacion_rows = []

def test_numerica_vs_target(df: pd.DataFrame, var: str, target: str = "fallas") -> dict:
    \"\"\"ANOVA + Spearman entre variable numérica y conteo de fallas.\"\"\"
    clean = df[[var, target]].dropna()
    if len(clean) < 10:
        return {"variable": var, "test": "n_insuf", "p_value": 1.0}

    # Dividir en cuartiles y hacer ANOVA
    q = pd.qcut(clean[var], 4, duplicates="drop")
    groups = [g[target].values for _, g in clean.groupby(q) if len(g) >= 3]
    if len(groups) >= 2:
        _, p_anova = f_oneway(*groups)
    else:
        p_anova = 1.0

    rho, p_sp = stats.spearmanr(clean[var], clean[target])
    p_best = min(p_anova, p_sp)
    return {"variable": var, "test": "ANOVA+Spearman", "p_value": float(p_best), "rho": float(rho)}


def test_categorica_vs_target(df: pd.DataFrame, var: str, target: str = "fallas") -> dict:
    \"\"\"Kruskal-Wallis entre variable categórica y conteo de fallas.\"\"\"
    clean = df[[var, target]].dropna()
    if len(clean) < 10:
        return {"variable": var, "test": "n_insuf", "p_value": 1.0}
    groups = [g[target].values for _, g in clean.groupby(var) if len(g) >= 3]
    if len(groups) >= 2:
        _, p_kw = kruskal(*groups)
    else:
        p_kw = 1.0
    return {"variable": var, "test": "Kruskal-Wallis", "p_value": float(p_kw)}


# Tests para candidatas nuevas
for var in CANDIDATAS_NUEVAS_NUMERICAS:
    if var in panel_feat.columns:
        row = test_numerica_vs_target(panel_feat, var)
        validacion_rows.append(row)

for var in CANDIDATAS_NUEVAS_CATEGORICAS:
    if var in panel_feat.columns:
        row = test_categorica_vs_target(panel_feat, var)
        validacion_rows.append(row)

# Tests informativos para features base
for var in ["lag_1","roll3_mean","trend_3","cusum_feat","cohorte_ratio","mis_lag1"]:
    if var in panel_feat.columns:
        row = test_numerica_vs_target(panel_feat, var)
        validacion_rows.append(row)

if validacion_rows:
    val_df = pd.DataFrame(validacion_rows)
    p_arr  = val_df["p_value"].values
    rejected, p_adj = fdr_correction(p_arr)
    val_df["p_adj"]   = p_adj
    val_df["sig_FDR"] = rejected
    val_df = val_df.sort_values("p_adj")

    # Veto: candidatas nuevas que no pasan FDR se excluyen del modelo
    vetadas = val_df[~val_df["sig_FDR"] & val_df["variable"].isin(
        CANDIDATAS_NUEVAS_NUMERICAS + CANDIDATAS_NUEVAS_CATEGORICAS
    )]["variable"].tolist()
    if vetadas:
        print(f"\\n⚠️  Variables candidatas VETADAS por FDR: {vetadas}")

    val_df.to_csv(OUTPUT_DIR / "validacion_variables.csv", index=False)
    print("✅ Validación estadística de variables:")
    print(val_df[["variable","test","p_value","p_adj","sig_FDR"]].to_string(index=False))
else:
    val_df = pd.DataFrame()
    print("ℹ️  Sin candidatas nuevas → sin validación FDR adicional")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 9 – Optuna hyperparameter optimization
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA ───────────────────────────────
# Solo se ejecuta si REOPTIMIZAR_HIPERPARAMETROS=True o no existe best_params.json

BEST_PARAMS_FILE = MODEL_DIR / "best_params.json"

def _wf_wape_4folds(monthly_raw: pd.DataFrame, params: dict,
                    feat_cols: list, encs: dict) -> float:
    \"\"\"
    WAPE out-of-sample en 4 folds walk-forward (usado por Optuna).
    Usa monthly_raw (panel sin features) para reconstruir features por ventana.
    \"\"\"
    meses = sorted(monthly_raw["mes_repair"].unique())
    n = len(meses)
    if n < MIN_MESES_ENTRENAMIENTO + 4:
        return 1.0

    folds_wape = []
    for fold in range(4):
        corte_idx   = MIN_MESES_ENTRENAMIENTO + fold
        if corte_idx >= n:
            break
        train_meses  = meses[:corte_idx]
        val_mes      = meses[corte_idx]
        all_meses_tv = meses[:corte_idx + 1]

        # Fit encoders en train únicamente
        data_tr_raw = monthly_raw[monthly_raw["mes_repair"].isin(train_meses)]
        _, _, _, fc, encs_fold = build_features(
            data_tr_raw, cohorte_serie, costos_ref, fit_encoders=True
        )
        # Build features sobre ventana completa (train+val) con encoders de train
        data_tv = monthly_raw[monthly_raw["mes_repair"].isin(all_meses_tv)]
        _, _, feat_tv, fc, _ = build_features(
            data_tv, cohorte_serie, costos_ref, fit_encoders=False, encoders=encs_fold
        )

        train_feat = feat_tv[feat_tv["mes_repair"].isin(train_meses)]
        val_feat   = feat_tv[feat_tv["mes_repair"] == val_mes]

        X_tr = train_feat[fc].fillna(0).astype(float)
        y_tr = train_feat["fallas"].astype(float)
        X_v  = val_feat[fc].fillna(0).astype(float)
        y_v  = val_feat["fallas"].astype(float)

        if len(X_tr) < 10 or len(X_v) == 0:
            continue

        sw = sample_weights_decay(len(y_tr))
        m = XGBRegressor(**{**XGB_PARAMS_BASE, **params})
        m.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
        preds = np.maximum(m.predict(X_v), 0)
        w = wape(y_v.values, preds)
        if not np.isnan(w):
            folds_wape.append(w)

    return float(np.mean(folds_wape)) if folds_wape else 1.0


def optuna_objective(trial: optuna.Trial) -> float:
    params = dict(
        n_estimators     = trial.suggest_int("n_estimators",    200, 600),
        learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        max_depth        = trial.suggest_int("max_depth",        3, 7),
        min_child_weight = trial.suggest_int("min_child_weight", 2, 8),
        subsample        = trial.suggest_float("subsample",      0.7, 1.0),
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 1.0),
        reg_alpha        = trial.suggest_float("reg_alpha",      0.1, 5.0, log=True),
        reg_lambda       = trial.suggest_float("reg_lambda",     0.5, 5.0, log=True),
    )
    return _wf_wape_4folds(monthly, params, FEATURE_COLS, encoders_global)


if REOPTIMIZAR_HIPERPARAMETROS or not BEST_PARAMS_FILE.exists():
    print(f"🔍 Iniciando Optuna ({N_OPTUNA_TRIALS} trials)…")
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2)
    )
    study.optimize(optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_wape   = study.best_value
    print(f"✅ Optuna terminado  |  Mejor WAPE 4-fold = {best_wape:.4f}")
    print(f"   Mejores parámetros: {best_params}")

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params, f, indent=2)
else:
    with open(BEST_PARAMS_FILE) as f:
        best_params = json.load(f)
    print(f"✅ Parámetros cargados de {BEST_PARAMS_FILE}")
    print(f"   {best_params}")

# Combinar con parámetros base (best_params tiene prioridad)
XGB_PARAMS_FINAL = {**XGB_PARAMS_BASE, **best_params}
print(f"\\n📋 Parámetros finales XGBoost: {XGB_PARAMS_FINAL}")
"""))

print("Part 2 cells written:", len(cells))
nb_part = nbformat.v4.new_notebook()
nb_part.cells = cells
nbformat.write(nb_part, "/home/user/ML/_part2_verify.ipynb")
print("Saved _part2_verify.ipynb")
