"""Part 5: Cells 16-18 (Score 7 components, Watchlist Top50, Power BI CSVs)"""
import nbformat

cells = []

# ─────────────────────────────────────────────────────────────────
# CELL 16 – 7-component score
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── SCORE DE ALERTA 7 COMPONENTES ───────────────────────────────────────────
# Cada componente se normaliza a [0,1] antes de ponderar.
# El score final es 0-100 (mayor = más urgente para el analista).

def normalizar_col(s: pd.Series) -> pd.Series:
    \"\"\"Min-max normalización robusta con percentil 99 como máximo.\"\"\"
    lo = s.min()
    hi = s.quantile(0.99)
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s.clip(lo, hi) - lo) / (hi - lo)).clip(0, 1)


def calcular_score(
    anomalia_df:    pd.DataFrame,
    IF_score:       pd.DataFrame,
    forecast_df:    pd.DataFrame,
    monthly:        pd.DataFrame,
    costos_ref:     pd.DataFrame,
    cohorte_serie:  pd.DataFrame,
    pesos:          dict,
    prev_forecast:  pd.DataFrame = None,
) -> pd.DataFrame:
    \"\"\"Ensambla el score de 7 componentes por serie_id.\"\"\"

    # ── Base: anomalía estadística (Repair Date) ─────────────────────────────
    base = anomalia_df[["serie_id","cusum_final","z_score_reciente",
                         "aceleracion_3m","fallas_mean_hist","fallas_max_hist",
                         "fallas_recientes","p_mw_adj","sig_mw"]].copy()

    # ── C1: Desviación estadística (z-score) ─────────────────────────────────
    base["c1_desviacion"] = normalizar_col(base["z_score_reciente"].clip(lower=0))

    # ── C2: Aceleración CUSUM ─────────────────────────────────────────────────
    base["c2_cusum"] = normalizar_col(base["cusum_final"].clip(lower=0))

    # ── C3: Señal Engine Date (cohorte_ratio) ─────────────────────────────────
    base = base.merge(cohorte_serie[["serie_id","cohorte_ratio_max"]].fillna(1.0),
                      on="serie_id", how="left")
    base["cohorte_ratio_max"] = base["cohorte_ratio_max"].fillna(1.0)
    # Señal anticipada: ratio > 1 indica cohortes fallando más de lo esperado
    base["c3_engine_date"] = normalizar_col((base["cohorte_ratio_max"] - 1.0).clip(lower=0))

    # ── C4: Anomalía Isolation Forest ────────────────────────────────────────
    base = base.merge(IF_score_serie, on="serie_id", how="left")
    base["if_score_reciente"] = base["if_score_reciente"].fillna(0)
    base["c4_if"] = normalizar_col(base["if_score_reciente"])

    # ── C5: Error forecast previo ────────────────────────────────────────────
    if prev_forecast is not None and len(prev_forecast) > 0:
        # Comparar forecast anterior vs realidad reciente
        ult_mes_hist = monthly.groupby("serie_id")["fallas"].last().rename("fallas_real")
        prev_f       = prev_forecast.groupby("serie_id")["fallas_central"].first().rename("fallas_pred_prev")
        err_df       = ult_mes_hist.to_frame().join(prev_f)
        err_df["error_rel"] = (np.abs(err_df["fallas_real"] - err_df["fallas_pred_prev"]) /
                               (err_df["fallas_real"].abs() + 1e-8))
        base = base.merge(err_df["error_rel"].reset_index(), on="serie_id", how="left")
        base["c5_error_prev"] = normalizar_col(base["error_rel"].fillna(0))
    else:
        base["c5_error_prev"] = 0.0

    # ── C6: Crecimiento proyectado ────────────────────────────────────────────
    # Promedio 6 meses proyectados vs promedio últimos 6 históricos
    hist_tail6 = (monthly.groupby("serie_id")
                  .apply(lambda g: g.sort_values("mes_repair")["fallas"].tail(6).mean())
                  .rename("mean_hist6").reset_index())
    fc_mean6   = (forecast_df.groupby("serie_id")["fallas_central"].mean()
                  .rename("mean_fc6").reset_index())
    growth_df  = hist_tail6.merge(fc_mean6, on="serie_id", how="left")
    growth_df["growth_ratio"] = (growth_df["mean_fc6"] /
                                 (growth_df["mean_hist6"].abs() + 1e-8))
    base = base.merge(growth_df[["serie_id","growth_ratio"]], on="serie_id", how="left")
    base["c6_crecimiento"] = normalizar_col((base["growth_ratio"].fillna(1) - 1).clip(lower=0))

    # ── C7: Costo relativo ───────────────────────────────────────────────────
    costo_total_serie = (monthly.groupby("serie_id")["costo_total"].sum()
                         .rename("costo_hist_total").reset_index())
    base = base.merge(costo_total_serie, on="serie_id", how="left")
    base["c7_costo"] = normalizar_col(base["costo_hist_total"].fillna(0))

    # ── Score ponderado ──────────────────────────────────────────────────────
    base["score"] = (
        pesos["desviacion_stat"]     * base["c1_desviacion"] +
        pesos["aceleracion_cusum"]   * base["c2_cusum"] +
        pesos["senal_engine_date"]   * base["c3_engine_date"] +
        pesos["anomalia_if"]         * base["c4_if"] +
        pesos["error_forecast_prev"] * base["c5_error_prev"] +
        pesos["crecimiento_proy"]    * base["c6_crecimiento"] +
        pesos["costo_relativo"]      * base["c7_costo"]
    ) * 100

    # ── Nivel de alerta ───────────────────────────────────────────────────────
    base["nivel_alerta"] = pd.cut(
        base["score"],
        bins   = [-1, 33, 66, 101],
        labels = ["BAJA", "MEDIA", "ALTA"]
    )

    # ── Razones explicativas (texto para el analista) ─────────────────────────
    def razones(row):
        r = []
        if row["c1_desviacion"] > 0.5:   r.append(f"Desv.stat={row['z_score_reciente']:.1f}σ")
        if row["c2_cusum"]      > 0.5:   r.append(f"CUSUM={row['cusum_final']:.1f}")
        if row["c3_engine_date"]> 0.5:   r.append(f"Cohorte={row['cohorte_ratio_max']:.2f}x")
        if row["c4_if"]         > 0.5:   r.append("IF=anomalía")
        if row.get("c5_error_prev", 0)>0.5: r.append("ErrFcPrev>50%")
        if row["c6_crecimiento"] > 0.5:  r.append(f"Crec={row.get('growth_ratio',1):.1f}x")
        if row["c7_costo"]      > 0.5:   r.append("CostoAlto")
        return " | ".join(r) if r else "Normal"

    base["razones"] = base.apply(razones, axis=1)

    return base.sort_values("score", ascending=False).reset_index(drop=True)


# Cargar forecast previo si existe (para c5)
prev_fc_file = MODEL_DIR / "forecast_historico.json"
prev_forecast_df = None
if prev_fc_file.exists():
    try:
        prev_forecast_df = pd.read_json(prev_fc_file)
    except Exception:
        prev_forecast_df = None

score_df = calcular_score(
    anomalia_df, IF_score_serie, forecast_df,
    monthly, costos_ref, cohorte_serie,
    PESOS_SCORE, prev_forecast_df
)

print(f"✅ Score calculado para {len(score_df)} series")
print(f"   ALTA  : {(score_df['nivel_alerta']=='ALTA').sum()}")
print(f"   MEDIA : {(score_df['nivel_alerta']=='MEDIA').sum()}")
print(f"   BAJA  : {(score_df['nivel_alerta']=='BAJA').sum()}")
print(f"\\n   Top 5 series por score:")
print(score_df[["serie_id","score","nivel_alerta","razones"]].head(5).to_string(index=False))
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 17 – Watchlist Top 50
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── WATCHLIST TOP 50 ─────────────────────────────────────────────────────────

# Combinar score con proyecciones y datos históricos
n_hist_serie = monthly.groupby("serie_id").agg(
    fallas_historicas = ("fallas",      "sum"),
    costo_historico   = ("costo_total", "sum"),
    primera_falla     = ("mes_dt",      "min"),
    ultima_falla      = ("mes_dt",      "max"),
).reset_index()

watchlist = (
    score_df
    .merge(costo_proy_6m,   on="serie_id", how="left")
    .merge(fallas_proy_6m,  on="serie_id", how="left")
    .merge(n_hist_serie,    on="serie_id", how="left")
    .merge(cohorte_serie[["serie_id","cohorte_ratio_max","cohorte_ratio_mean"]],
           on="serie_id", how="left")
)

watchlist["costo_proj_6m"]  = watchlist["costo_proj_6m"].fillna(0)
watchlist["fallas_proj_6m"] = watchlist["fallas_proj_6m"].fillna(0)

# Extraer componentes del serie_id para columnas independientes
watchlist[["ea_number_w","componente_w","damage_code_w"]] = (
    watchlist["serie_id"].str.split("||", expand=True).iloc[:, :3]
)

# Top 50 por score
top50 = watchlist.head(50).copy()

# Añadir columnas c_* para el dashboard de Power BI
for col in ["c1_desviacion","c2_cusum","c3_engine_date","c4_if",
            "c5_error_prev","c6_crecimiento","c7_costo"]:
    if col not in top50.columns:
        top50[col] = 0.0

top50 = top50.rename(columns={
    "ea_number_w": "ea_number",
    "componente_w": "componente_pw",
    "damage_code_w": "damage_code_pw",
    "cohorte_ratio_max": "cohorte_ratio",
})

print(f"✅ Watchlist Top 50 generada")
print(top50[["serie_id","score","nivel_alerta","fallas_proj_6m",
             "costo_proj_6m","razones"]].head(10).to_string(index=False))
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 18 – Power BI CSV exports (8 files)
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── EXPORTAR 8 CSVs PARA POWER BI ───────────────────────────────────────────
# Todas las rutas son relativas al directorio de trabajo (Path())

fecha_calc = datetime.now().strftime("%Y-%m-%d %H:%M")

# ── 1. watchlist_top50.csv ────────────────────────────────────────────────────
cols_top50 = [
    "serie_id","componente_pw","damage_code_pw","mis_cluster","score","nivel_alerta",
    "razones","cohorte_ratio","costo_proj_6m","fallas_proj_6m",
    "c1_desviacion","c2_cusum","c3_engine_date","c4_if",
    "c5_error_prev","c6_crecimiento","c7_costo",
]
cols_top50 = [c for c in cols_top50 if c in top50.columns]
top50[cols_top50].to_csv(OUTPUT_DIR / "watchlist_top50.csv", index=False)

# ── 2. componentes_nuevos.csv ─────────────────────────────────────────────────
# Series con historia < 6 meses → requieren revisión manual del analista
nuevos_mask = (n_hist_serie["fallas_historicas"] > 0) & \
              (monthly.groupby("serie_id").size().reindex(n_hist_serie["serie_id"]).fillna(0).values < MIN_MESES_SERIE)
comp_nuevos = n_hist_serie[nuevos_mask].copy()
comp_nuevos.to_csv(OUTPUT_DIR / "componentes_nuevos.csv", index=False)

# ── 3. tendencias_forecast.csv ────────────────────────────────────────────────
# Histórico + forecast ML con bandas (para gráficos de tendencia)
hist_tend = monthly[["serie_id","mes_dt","fallas","costo_total"]].copy()
hist_tend["tipo"]         = "Histórico"
hist_tend["fallas_lo"]    = hist_tend["fallas"]
hist_tend["fallas_hi"]    = hist_tend["fallas"]
hist_tend["costo_central"]= hist_tend["costo_total"]
hist_tend["costo_lo"]     = hist_tend["costo_total"]
hist_tend["costo_hi"]     = hist_tend["costo_total"]
hist_tend = hist_tend.rename(columns={"fallas": "fallas_central"})

fc_tend = forecast_df[["serie_id","mes_dt","fallas_central","fallas_lo","fallas_hi",
                        "costo_central","costo_lo","costo_hi"]].copy()
fc_tend["tipo"] = "Forecast ML"

tendencias = pd.concat([
    hist_tend[["serie_id","mes_dt","tipo","fallas_central","fallas_lo","fallas_hi",
               "costo_central","costo_lo","costo_hi"]],
    fc_tend
], ignore_index=True).sort_values(["serie_id","mes_dt"])
tendencias.to_csv(OUTPUT_DIR / "tendencias_forecast.csv", index=False)

# ── 4. heatmap_aceleracion.csv ────────────────────────────────────────────────
top20_series = top50["serie_id"].head(20).tolist()
heatmap_df = (
    monthly[monthly["serie_id"].isin(top20_series)]
    [["serie_id","mes_dt","fallas"]]
    .rename(columns={"mes_dt":"mes_repair","fallas":"fallas"})
    .copy()
)
heatmap_df["componente"] = heatmap_df["serie_id"].str.split("||").str[1]
heatmap_df["damage_code"]= heatmap_df["serie_id"].str.split("||").str[2]
heatmap_df.to_csv(OUTPUT_DIR / "heatmap_aceleracion.csv", index=False)

# ── 5. serie_mensual_completa.csv ─────────────────────────────────────────────
cols_fact = ["serie_id","mes_dt","fallas","costo_total","mis_mean","km_mean",
             "ea_number","componente","damage_code","mis_cluster"]
cols_fact = [c for c in cols_fact if c in monthly.columns]
monthly[cols_fact].to_csv(OUTPUT_DIR / "serie_mensual_completa.csv", index=False)

# ── 6. score_diagnostico.csv ─────────────────────────────────────────────────
diag_cols = ["serie_id","score","nivel_alerta","c1_desviacion","c2_cusum",
             "c3_engine_date","c4_if","c5_error_prev","c6_crecimiento","c7_costo",
             "razones","cusum_final","z_score_reciente","aceleracion_3m",
             "cohorte_ratio_max","if_score_reciente"]
diag_cols = [c for c in diag_cols if c in score_df.columns]
score_diag = score_df[diag_cols].copy()
score_diag["fecha_calculo"] = fecha_calc
score_diag.to_csv(OUTPUT_DIR / "score_diagnostico.csv", index=False)

# ── 7. validacion_variables.csv ───────────────────────────────────────────────
if len(val_df) > 0:
    val_df["fecha_calculo"] = fecha_calc
    val_df.to_csv(OUTPUT_DIR / "validacion_variables.csv", index=False)

# ── 8. historial_accuracy.csv ────────────────────────────────────────────────
acc_cols = ["fold","mes_val","n_train","n_val","wape_oos","wape_is","gap_of","bias_oos","mae_oos"]
acc_cols = [c for c in acc_cols if c in wf_results.columns]
acc_df = wf_results[acc_cols].copy()
acc_df["fecha_calculo"] = fecha_calc
acc_df["modo"] = MODO
acc_df.to_csv(OUTPUT_DIR / "historial_accuracy.csv", index=False)

# ── Guardar forecast histórico para comparar en el próximo run ───────────────
forecast_df.to_json(MODEL_DIR / "forecast_historico.json", orient="records")

# ── Guardar metadata del modelo ───────────────────────────────────────────────
metadata = {
    "version":         "v6",
    "fecha":           fecha_calc,
    "modo":            MODO,
    "n_series":        int(monthly["serie_id"].nunique()),
    "wape_oos_mean":   float(wf_results["wape_oos"].mean()) if len(wf_results) else None,
    "wape_is_mean":    float(wf_results["wape_is"].mean())  if len(wf_results) else None,
    "gap_of_mean":     float(wf_results["gap_of"].mean())   if len(wf_results) else None,
    "n_folds":         int(len(wf_results)),
    "horizonte_meses": HORIZONTE_MESES,
    "features":        FEATURE_COLS,
    "xgb_params":      XGB_PARAMS_FINAL,
}
with open(MODEL_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)

print("✅ 8 CSVs exportados a", OUTPUT_DIR)
for fp in sorted(OUTPUT_DIR.glob("*.csv")):
    n = len(pd.read_csv(fp))
    print(f"   {fp.name:40s} {n:6,} filas")
"""))

print("Part 5 cells written:", len(cells))
nb_part = nbformat.v4.new_notebook()
nb_part.cells = cells
nbformat.write(nb_part, "/home/user/ML/_part5_verify.ipynb")
print("Saved _part5_verify.ipynb")
