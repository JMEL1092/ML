"""Part 3: Cells 10-12 (Walk-forward backtesting, Global model, Individual models)"""
import nbformat

cells = []

# ─────────────────────────────────────────────────────────────────
# CELL 10 – Walk-forward backtesting
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── BACKTESTING WALK-FORWARD ─────────────────────────────────────────────────
# Ventana de entrenamiento mínima: 12 meses
# Paso: 1 mes | Mínimo 6 folds
# WAPE reportado por fold (in-sample y OOS) para detectar overfitting.

def walk_forward_validation(monthly_raw: pd.DataFrame, feat_cols: list,
                             params: dict) -> tuple:
    \"\"\"
    Walk-forward sobre el panel MENSUAL RAW.
    Para cada fold:
      1. Fit encoders sólo en train_meses
      2. Build features sobre (train+val) con esos encoders
      3. Entrenar en train, predecir en val → WAPE OOS
    \"\"\"
    meses    = sorted(monthly_raw["mes_repair"].unique())
    n_meses  = len(meses)
    max_folds = n_meses - MIN_MESES_ENTRENAMIENTO

    if max_folds < WF_FOLDS_MINIMOS:
        print(f"⚠️  Solo {max_folds} folds disponibles (mínimo requerido: {WF_FOLDS_MINIMOS})")

    resultados    = []
    all_residuals = []

    for fold in range(max_folds):
        corte_idx    = MIN_MESES_ENTRENAMIENTO + fold
        train_meses  = meses[:corte_idx]
        val_mes      = meses[corte_idx]
        all_meses_tv = meses[:corte_idx + 1]

        # Fit encoders SÓLO en datos de entrenamiento
        data_tr_only = monthly_raw[monthly_raw["mes_repair"].isin(train_meses)]
        _, _, _, fc, encs_fold = build_features(
            data_tr_only, cohorte_serie, costos_ref, fit_encoders=True
        )

        # Build features en ventana completa (train+val) con encoders de train
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

        if len(X_tr) < 10 or len(X_v) == 0 or y_v.sum() == 0:
            continue

        # Early stopping con últimas 3 meses de train como eval set
        n_series   = monthly_raw["serie_id"].nunique()
        eval_rows  = min(3 * n_series, max(len(X_tr) // 6, 5))
        X_eval, y_eval = X_tr.iloc[-eval_rows:], y_tr.iloc[-eval_rows:]
        X_fit,  y_fit  = X_tr.iloc[:-eval_rows], y_tr.iloc[:-eval_rows]
        sw_fit = sample_weights_decay(len(y_fit))

        if len(X_fit) < 5:
            X_fit, y_fit = X_tr, y_tr
            sw_fit = sample_weights_decay(len(y_fit))
            X_eval, y_eval = X_tr.iloc[-2:], y_tr.iloc[-2:]

        model = XGBRegressor(**{**params, "early_stopping_rounds": 30, "eval_metric": "mae"})
        model.fit(X_fit, y_fit, sample_weight=sw_fit,
                  eval_set=[(X_eval, y_eval)], verbose=False)

        preds_tr  = np.maximum(model.predict(X_tr), 0)
        wape_is   = wape(y_tr.values, preds_tr)
        preds_oos = np.maximum(model.predict(X_v),  0)
        wape_oos  = wape(y_v.values, preds_oos)
        bias_oos  = float(np.mean(preds_oos - y_v.values))
        mae_oos   = float(np.mean(np.abs(preds_oos - y_v.values)))

        gap = (wape_is - wape_oos) if not (np.isnan(wape_is) or np.isnan(wape_oos)) else np.nan
        all_residuals.extend((y_v.values - preds_oos).tolist())

        resultados.append({
            "fold": fold + 1, "mes_val": str(val_mes),
            "n_train": len(X_tr), "n_val": len(X_v),
            "wape_is": wape_is, "wape_oos": wape_oos,
            "gap_of": gap, "bias_oos": bias_oos, "mae_oos": mae_oos,
            "n_est_eff": model.best_iteration if model.best_iteration else params.get("n_estimators", 300),
        })

        if (fold + 1) % 3 == 0:
            print(f"  Fold {fold+1:2d} | {val_mes} | WAPE OOS={wape_oos:.4f} IS={wape_is:.4f} gap={gap:+.4f}")

    return pd.DataFrame(resultados), all_residuals


print("Ejecutando walk-forward backtesting…")
wf_results, wf_residuals = walk_forward_validation(monthly, FEATURE_COLS, XGB_PARAMS_FINAL)

print(f"\\n{'='*60}")
print("RESULTADOS WALK-FORWARD")
print(f"{'='*60}")
print(wf_results[["fold","mes_val","n_val","wape_oos","wape_is","gap_of","bias_oos"]].to_string(index=False))

wape_oos_mean  = wf_results["wape_oos"].mean()
wape_is_mean   = wf_results["wape_is"].mean()
gap_mean       = wf_results["gap_of"].mean()
n_folds_ok     = (wf_results["wape_oos"] <= WAPE_MAX_OOS).sum()
n_folds_total  = len(wf_results)

print(f"\\n  WAPE OOS promedio : {wape_oos_mean:.4f}  (límite {WAPE_MAX_OOS})")
print(f"  WAPE IS  promedio : {wape_is_mean:.4f}")
print(f"  Gap promedio       : {gap_mean:+.4f}  (límite {GAP_OVERFITTING_MAX})")
print(f"  Folds WAPE ≤ 11%  : {n_folds_ok}/{n_folds_total}")

# Calcular cobertura CI 95%
if wf_residuals:
    sigma_wf = np.std(wf_residuals)
    # Cobertura: qué % de residuos caben dentro de ±1.96σ
    within = np.mean(np.abs(wf_residuals) <= 1.96 * sigma_wf)
    print(f"  Cobertura CI 95%  : {within:.3f}  (mínimo {COBERTURA_CI_MIN})")
else:
    within = 0.0

# Guardar métricas de backtesting
wf_results.to_csv(OUTPUT_DIR / "historial_accuracy.csv", index=False)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 11 – Global XGBoost model
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── MODELO GLOBAL XGBOOST ────────────────────────────────────────────────────
# Entrena sobre TODO el panel histórico con los hiperparámetros finales.

print("Entrenando modelo global XGBoost…")

# Re-build features sobre dataset completo con encoders globales
X_all, y_all, panel_feat_full, FEATURE_COLS, encoders_global = build_features(
    monthly, cohorte_serie, costos_ref, fit_encoders=True
)

sw_all = sample_weights_decay(len(y_all))

# Eval set: últimos 3 meses de datos
meses_all = sorted(panel_feat_full["mes_repair"].unique())
eval_meses = meses_all[-3:]
mask_eval  = panel_feat_full["mes_repair"].isin(eval_meses)

X_eval_g  = X_all[mask_eval].values
y_eval_g  = y_all[mask_eval].values
X_fit_g   = X_all[~mask_eval].values
y_fit_g   = y_all[~mask_eval].values
sw_fit    = sample_weights_decay(len(y_fit_g))

model_global = XGBRegressor(**{**XGB_PARAMS_FINAL, "early_stopping_rounds": 30, "eval_metric": "mae"})
model_global.fit(
    X_fit_g, y_fit_g,
    sample_weight = sw_fit,
    eval_set      = [(X_eval_g, y_eval_g)],
    verbose       = False
)

n_est_eff_global = model_global.best_iteration if model_global.best_iteration else XGB_PARAMS_FINAL["n_estimators"]
preds_tr_global  = np.maximum(model_global.predict(X_all.values), 0)
wape_is_global   = wape(y_all.values, preds_tr_global)

print(f"✅ Modelo global entrenado")
print(f"   n_estimators efectivos : {n_est_eff_global}")
print(f"   WAPE in-sample          : {wape_is_global:.4f}")

# Importancia de features
feat_imp = pd.Series(model_global.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print(f"\\n   Top-10 features por importancia:")
print(feat_imp.head(10).to_string())

# Guardar modelo global
joblib.dump(model_global, MODEL_DIR / "global_xgb.pkl")
joblib.dump(encoders_global, MODEL_DIR / "encoders.pkl")
print(f"\\n💾 Modelo global guardado en {MODEL_DIR}/global_xgb.pkl")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 12 – Individual models per series
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── MODELOS INDIVIDUALES POR SERIE ───────────────────────────────────────────
# Series con ≥ 12 meses → early stopping | < 12 meses → n_estimators=100 fijo

def entrenar_modelo_individual(serie_id: str, g: pd.DataFrame,
                                costos_ref: pd.DataFrame,
                                cohorte_serie: pd.DataFrame) -> dict:
    \"\"\"
    Entrena XGBoost individual para una serie.
    Retorna dict con modelo, n_est_efectivo, wape_is, wape_oos_local.
    \"\"\"
    g = g.sort_values("mes_repair").copy()
    n_meses = len(g)

    if n_meses < MIN_MESES_SERIE:
        return None

    # Features individuales (encoders propios por serie)
    panel_s = g.copy()
    _, _, feat_s, fc_s, enc_s = build_features(
        panel_s, cohorte_serie, costos_ref, fit_encoders=True
    )
    X_s = feat_s[fc_s].fillna(0).astype(float).values
    y_s = feat_s["fallas"].astype(float).values

    if len(X_s) < 4:
        return None

    sw_s = sample_weights_decay(len(y_s))

    if n_meses >= MIN_MESES_ENTRENAMIENTO:
        # Early stopping con últimos 3 puntos como eval
        X_ev, y_ev = X_s[-3:], y_s[-3:]
        X_fi, y_fi = X_s[:-3], y_s[:-3]
        if len(X_fi) < 4:
            X_fi, y_fi = X_s, y_s
            X_ev, y_ev = X_s[-2:], y_s[-2:]
        sw_fi = sample_weights_decay(len(y_fi))
        m = XGBRegressor(**{**XGB_PARAMS_FINAL, "early_stopping_rounds": 30, "eval_metric": "mae"})
        m.fit(X_fi, y_fi, sample_weight=sw_fi, eval_set=[(X_ev, y_ev)], verbose=False)
        n_eff = m.best_iteration if m.best_iteration else XGB_PARAMS_FINAL["n_estimators"]
    else:
        # n_estimators fijo para series cortas
        params_short = {**XGB_PARAMS_FINAL, "n_estimators": 100}
        params_short.pop("early_stopping_rounds", None)
        m = XGBRegressor(**params_short)
        m.fit(X_s, y_s, sample_weight=sw_s, verbose=False)
        n_eff = 100

    preds_s   = np.maximum(m.predict(X_s), 0)
    wape_s    = wape(y_s, preds_s)

    return {
        "model":   m,
        "n_eff":   n_eff,
        "wape_is": wape_s,
        "n_meses": n_meses,
        "encoders": enc_s,
        "fc":       fc_s,
    }


# Entrenar modelos individuales para series con datos suficientes
modelos_individuales = {}
series_validas = panel_feat_full.groupby("serie_id").filter(
    lambda g: len(g) >= MIN_MESES_SERIE
)["serie_id"].unique()

n_ind = 0
wape_ind_list = []
for sid in series_validas:
    g_s = monthly[monthly["serie_id"] == sid]
    res = entrenar_modelo_individual(sid, g_s, costos_ref, cohorte_serie)
    if res is not None:
        modelos_individuales[sid] = res
        joblib.dump(res, MODEL_DIR / "individuales" / f"{hash_serie(sid)}.pkl")
        wape_ind_list.append(res["wape_is"])
        n_ind += 1

print(f"✅ Modelos individuales entrenados: {n_ind}")
if wape_ind_list:
    print(f"   WAPE IS medio (individuales): {np.mean(wape_ind_list):.4f}")
    print(f"   n_estimators efectivo medio : {np.mean([r['n_eff'] for r in modelos_individuales.values()]):.0f}")
"""))

print("Part 3 cells written:", len(cells))
nb_part = nbformat.v4.new_notebook()
nb_part.cells = cells
nbformat.write(nb_part, "/home/user/ML/_part3_verify.ipynb")
print("Saved _part3_verify.ipynb")
