"""Part 4: Cells 13-15 (6M Forecast, Isolation Forest, CUSUM)"""
import nbformat

cells = []

# ─────────────────────────────────────────────────────────────────
# CELL 13 – 6-month recursive forecast
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── FORECAST RECURSIVO 6 MESES ───────────────────────────────────────────────
# Usa modelo individual si existe, sino modelo global.
# El cap_forecast se aplica en cada paso con historia_real (nunca predicciones).

def forecast_serie(serie_id: str, monthly_panel: pd.DataFrame,
                   model_global: object, modelos_ind: dict,
                   feat_cols: list, encoders: dict,
                   costos_ref: pd.DataFrame, cohorte_serie: pd.DataFrame,
                   wf_residuals: list,
                   n_meses: int = HORIZONTE_MESES) -> pd.DataFrame:
    \"\"\"Genera forecast recursivo para una serie. Retorna DataFrame de predicciones.\"\"\"
    g = monthly_panel[monthly_panel["serie_id"] == serie_id].sort_values("mes_repair").copy()
    if len(g) == 0:
        return pd.DataFrame()

    historia_real = g["fallas"].values.copy()   # solo valores reales
    ultimo_mes    = g["mes_repair"].iloc[-1]
    ultimo_dt     = g["mes_dt"].iloc[-1]

    # Metadatos estáticos de la serie
    ea       = g["ea_number"].iloc[-1]
    comp     = g["componente"].iloc[-1]
    dc       = g["damage_code"].iloc[-1]
    mis_cl   = g["mis_cluster"].iloc[-1]

    # Contexto de costo
    costo_ref_row = costos_ref[costos_ref["serie_id"] == serie_id]
    costo_unit    = float(costo_ref_row["costo_unit_med"].values[0]) if len(costo_ref_row) else 0.0

    # Seleccionar modelo
    usar_individual = serie_id in modelos_ind
    if usar_individual:
        m_ind      = modelos_ind[serie_id]["model"]
        enc_ind    = modelos_ind[serie_id]["encoders"]
        fc_ind     = modelos_ind[serie_id]["fc"]
    residuals_arr = np.array(wf_residuals) if wf_residuals else np.array([0.0])

    predicciones = []
    # Estado deslizante para el forecast recursivo (no se mezcla con historia_real)
    ventana      = g.copy()

    for h in range(n_meses):
        sig_mes_period = ultimo_mes + (h + 1)
        sig_mes_dt     = sig_mes_period.to_timestamp()

        # Construir fila de features para el siguiente mes
        _, _, feat_next, fc_g, _ = build_features(
            ventana, cohorte_serie, costos_ref, fit_encoders=False, encoders=encoders
        )
        if len(feat_next) == 0:
            break

        row_feat = feat_next[fc_g].iloc[[-1]].fillna(0).astype(float)

        if usar_individual:
            if fc_ind == fc_g or all(c in feat_next.columns for c in fc_ind):
                row_ind = feat_next[fc_ind].iloc[[-1]].fillna(0).astype(float)
                pred_i  = float(np.maximum(m_ind.predict(row_ind.values), 0)[0])
            else:
                pred_i  = float(np.maximum(model_global.predict(row_feat.values), 0)[0])
            pred_g   = float(np.maximum(model_global.predict(row_feat.values), 0)[0])
            # Blend: 60% individual, 40% global
            pred_raw = 0.6 * pred_i + 0.4 * pred_g
        else:
            pred_raw = float(np.maximum(model_global.predict(row_feat.values), 0)[0])

        # Cap conservador: usa SÓLO historia_real
        pred_cap = cap_forecast(pred_raw, historia_real)

        lo, hi = ci_bandas(np.array([pred_cap] * (h + 1)), residuals_arr)
        lo_h, hi_h = float(lo[-1]), float(hi[-1])

        predicciones.append({
            "serie_id":      serie_id,
            "ea_number":     ea,
            "componente":    comp,
            "damage_code":   dc,
            "mis_cluster":   mis_cl,
            "mes_forecast":  str(sig_mes_period),
            "mes_dt":        sig_mes_dt,
            "horizonte":     h + 1,
            "fallas_central": pred_cap,
            "fallas_lo":     lo_h,
            "fallas_hi":     hi_h,
            "costo_central": pred_cap * costo_unit,
            "costo_lo":      lo_h  * costo_unit,
            "costo_hi":      hi_h  * costo_unit,
            "modelo_usado":  "blend" if usar_individual else "global",
        })

        # Agregar fila sintética a la ventana para el siguiente paso recursivo
        nueva_fila = ventana.iloc[[-1]].copy()
        nueva_fila["mes_repair"] = sig_mes_period
        nueva_fila["mes_dt"]     = sig_mes_dt
        nueva_fila["fallas"]     = pred_cap          # se usa para lag en h+2
        nueva_fila["costo_total"]= pred_cap * costo_unit
        ventana = pd.concat([ventana, nueva_fila], ignore_index=True)
        # NOTA: historia_real permanece inalterada (solo datos reales)

    return pd.DataFrame(predicciones)


# Generar forecasts para TODAS las series con suficientes datos
print("Generando forecast 6 meses para todas las series…")
forecast_rows = []

series_forecast = monthly["serie_id"].unique()
for i, sid in enumerate(series_forecast):
    g_hist = monthly[monthly["serie_id"] == sid]
    if g_hist["fallas"].sum() == 0 or len(g_hist) < 2:
        continue
    fc_df = forecast_serie(
        sid, monthly, model_global, modelos_individuales,
        FEATURE_COLS, encoders_global, costos_ref, cohorte_serie, wf_residuals
    )
    forecast_rows.append(fc_df)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(series_forecast)} series procesadas…")

forecast_df = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
print(f"\\n✅ Forecast generado: {len(forecast_df):,} filas ({forecast_df['serie_id'].nunique()} series × 6 meses)")

# Verificar cap: forecast mes 6 ≤ 1.5× máximo histórico real por serie
violaciones_cap = 0
for sid, g_fc in forecast_df[forecast_df["horizonte"] == 6].groupby("serie_id"):
    hist_max = monthly[monthly["serie_id"] == sid]["fallas"].max()
    fc6 = g_fc["fallas_central"].iloc[0]
    if fc6 > hist_max * 1.5 + 0.01:
        violaciones_cap += 1
print(f"   Violaciones de cap (fc_M6 > 1.5× hist_max): {violaciones_cap}")

# Proyección de costo total a 6 meses por serie
costo_proy_6m = (forecast_df.groupby("serie_id")["costo_central"].sum()
                 .rename("costo_proj_6m").reset_index())
fallas_proy_6m = (forecast_df.groupby("serie_id")["fallas_central"].sum()
                  .rename("fallas_proj_6m").reset_index())
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 14 – Isolation Forest
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── ISOLATION FOREST (ANOMALÍAS MULTIVARIADAS) ───────────────────────────────
# Entrena sobre la matriz de features del histórico completo.
# Detecta combinaciones inusuales de lag/rolling/cohorte.

IF_FEATURES = [
    "lag_1","lag_2","lag_3","lag_6",
    "roll3_mean","roll6_mean","roll3_cost","roll6_cost",
    "trend_3","trend_6","cusum_feat",
    "mis_lag1","km_lag1","cohorte_ratio",
]
# Solo columnas que existen en panel
IF_FEATURES = [c for c in IF_FEATURES if c in panel_feat_full.columns]

X_if = panel_feat_full[IF_FEATURES].fillna(0).astype(float)

scaler_if = StandardScaler()
X_if_sc   = scaler_if.fit_transform(X_if)

iso_forest = IsolationForest(
    n_estimators = 200,
    contamination = 0.05,
    random_state  = 42,
    n_jobs        = -1,
)
iso_forest.fit(X_if_sc)

# Score de anomalía: más negativo = más anómalo → invertir y normalizar 0-1
raw_scores = iso_forest.score_samples(X_if_sc)
# Escalar a [0, 1] donde 1 = más anómalo
if_score_norm = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-10)

panel_feat_full["if_score"] = if_score_norm

# Agregar IF score por serie (máximo de sus observaciones recientes)
IF_score_serie = (
    panel_feat_full.sort_values("mes_repair")
    .groupby("serie_id")
    .apply(lambda g: g["if_score"].tail(3).mean())
    .rename("if_score_reciente")
    .reset_index()
)

# Guardar modelo IF
joblib.dump({"model": iso_forest, "scaler": scaler_if, "features": IF_FEATURES},
            MODEL_DIR / "isolation_forest.pkl")

n_anomalos = (if_score_norm > 0.7).sum()
print(f"✅ Isolation Forest entrenado  |  {n_anomalos:,} observaciones con score > 0.7")
print(f"   Features usadas: {IF_FEATURES}")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 15 – CUSUM + statistical detection
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── CUSUM Y DETECCIÓN ESTADÍSTICA ───────────────────────────────────────────
# Calcula por serie:
#   1. cusum_final       – CUSUM acumulado positivo normalizado
#   2. z_score_reciente  – desviación estadística del período reciente vs histórico
#   3. aceleracion_3m    – pendiente de los últimos 3 meses

def detectar_anomalias_serie(g: pd.DataFrame) -> dict:
    \"\"\"Calcula métricas de anomalía para una serie mensual.\"\"\"
    g = g.sort_values("mes_repair").copy()
    fallas_arr = g["fallas"].values.astype(float)
    n = len(fallas_arr)

    # CUSUM
    cs = cusum_stat(fallas_arr) if n >= 3 else 0.0

    # Z-score: media últimos 3 meses vs media histórica completa
    if n >= 6:
        mu_hist   = float(np.mean(fallas_arr[:-3]))
        sigma_hist= float(np.std(fallas_arr[:-3])) + 1e-8
        mu_rec    = float(np.mean(fallas_arr[-3:]))
        z         = (mu_rec - mu_hist) / sigma_hist
    else:
        z = 0.0

    # Aceleración (pendiente últimos 3 meses)
    accel = slope_lineal(fallas_arr[-3:]) if n >= 3 else 0.0

    # Mann-Whitney: últimos 3 meses vs histórico anterior (solo si n ≥ 8)
    if n >= 8:
        try:
            _, p_mw = mannwhitneyu(fallas_arr[-3:], fallas_arr[:-3], alternative="greater")
        except Exception:
            p_mw = 1.0
    else:
        p_mw = 1.0

    return {
        "cusum_final":       cs,
        "z_score_reciente":  z,
        "aceleracion_3m":    accel,
        "p_mw":              p_mw,
        "fallas_mean_hist":  float(np.mean(fallas_arr)),
        "fallas_max_hist":   float(np.max(fallas_arr)),
        "fallas_recientes":  float(np.mean(fallas_arr[-3:])) if n >= 3 else float(fallas_arr[-1]),
    }


anomalia_rows = []
for sid, g_s in monthly.groupby("serie_id"):
    res = detectar_anomalias_serie(g_s)
    res["serie_id"] = sid
    anomalia_rows.append(res)

anomalia_df = pd.DataFrame(anomalia_rows)

# Corrección FDR sobre los p-valores de Mann-Whitney
if "p_mw" in anomalia_df.columns:
    rejected_mw, p_adj_mw = fdr_correction(anomalia_df["p_mw"].values)
    anomalia_df["p_mw_adj"] = p_adj_mw
    anomalia_df["sig_mw"]   = rejected_mw
else:
    anomalia_df["sig_mw"] = False

print(f"✅ Detección estadística completada para {len(anomalia_df)} series")
print(f"   Series con CUSUM > 2   : {(anomalia_df['cusum_final'] > 2).sum()}")
print(f"   Series con Z-score > 2 : {(anomalia_df['z_score_reciente'] > 2).sum()}")
print(f"   Series sig. Mann-Whitney (FDR): {anomalia_df['sig_mw'].sum()}")
"""))

print("Part 4 cells written:", len(cells))
nb_part = nbformat.v4.new_notebook()
nb_part.cells = cells
nbformat.write(nb_part, "/home/user/ML/_part4_verify.ipynb")
print("Saved _part4_verify.ipynb")
