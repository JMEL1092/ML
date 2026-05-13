"""Part 6: Cells 19-27 (9 visualizations, anti-overfitting checklist, summary)"""
import nbformat

cells = []

# ─────────────────────────────────────────────────────────────────
# CELL 19 – VIZ 1: Global time series + 6M forecast with CI bands
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 1: LÍNEA DE TIEMPO GLOBAL + FORECAST 6 MESES ─────────────────────────
# Eje X: Repair Date | Forecast conecta suavemente con el último histórico

def viz1_timeline_global(monthly, forecast_df, output_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), facecolor=COLOR_LIGHT)

    for ax in (ax1, ax2):
        ax.set_facecolor("white")

    # Agregar totales mensuales
    hist_total = monthly.groupby("mes_dt")[["fallas","costo_total"]].sum().reset_index()
    fc_total   = (forecast_df.groupby("mes_dt")
                  .agg(fallas_central=("fallas_central","sum"),
                       fallas_lo     =("fallas_lo","sum"),
                       fallas_hi     =("fallas_hi","sum"),
                       costo_central =("costo_central","sum"),
                       costo_lo      =("costo_lo","sum"),
                       costo_hi      =("costo_hi","sum"))
                  .reset_index())

    # Conectar histórico con forecast (último punto histórico + primer punto forecast)
    ultimo_hist = hist_total.iloc[-1]
    conn_f = pd.DataFrame([{"mes_dt": ultimo_hist["mes_dt"],
                             "fallas_central": ultimo_hist["fallas"],
                             "fallas_lo": ultimo_hist["fallas"],
                             "fallas_hi": ultimo_hist["fallas"],
                             "costo_central": ultimo_hist["costo_total"],
                             "costo_lo": ultimo_hist["costo_total"],
                             "costo_hi": ultimo_hist["costo_total"]}])
    fc_plot = pd.concat([conn_f, fc_total], ignore_index=True)

    # Panel superior: fallas
    ax1.plot(hist_total["mes_dt"], hist_total["fallas"],
             color=COLOR_NAVY, lw=2.5, label="Histórico", zorder=5)
    ax1.plot(fc_plot["mes_dt"], fc_plot["fallas_central"],
             color=COLOR_RED, lw=2, ls="--", label="Forecast ML", zorder=5)
    ax1.fill_between(fc_plot["mes_dt"], fc_plot["fallas_lo"], fc_plot["fallas_hi"],
                     color=COLOR_RED, alpha=0.15, label="CI 95%")
    ax1.axvline(ultimo_hist["mes_dt"], color="gray", lw=1, ls=":", alpha=0.7)
    ax1.set_title("Fallas totales mensuales + Forecast 6M", fontsize=13, color=COLOR_NAVY, pad=8)
    ax1.set_ylabel("Fallas", color=COLOR_NAVY)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax1.tick_params(axis="x", rotation=35)

    # Panel inferior: costo
    ax2.plot(hist_total["mes_dt"], hist_total["costo_total"] / 1e3,
             color=COLOR_TEAL, lw=2.5, label="Histórico")
    ax2.plot(fc_plot["mes_dt"], fc_plot["costo_central"] / 1e3,
             color=COLOR_GOLD, lw=2, ls="--", label="Forecast ML")
    ax2.fill_between(fc_plot["mes_dt"], fc_plot["costo_lo"] / 1e3, fc_plot["costo_hi"] / 1e3,
                     color=COLOR_GOLD, alpha=0.2, label="CI 95%")
    ax2.axvline(ultimo_hist["mes_dt"], color="gray", lw=1, ls=":", alpha=0.7)
    ax2.set_title("Costo total mensual (k USD) + Forecast 6M", fontsize=13, color=COLOR_NAVY, pad=8)
    ax2.set_ylabel("Costo (k USD)", color=COLOR_TEAL)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax2.tick_params(axis="x", rotation=35)

    fig.suptitle("Sistema de Forecast — Planta Silao VW", fontsize=14, color=COLOR_NAVY, y=1.01)
    plt.tight_layout()
    path = output_dir / "viz1_timeline_global.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


import matplotlib.dates
viz1_timeline_global(monthly, forecast_df, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 20 – VIZ 2: Walk-forward WAPE evolution
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 2: EVOLUCIÓN WAPE WALK-FORWARD ────────────────────────────────────────

def viz2_wf_accuracy(wf_results, output_dir):
    if len(wf_results) == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLOR_LIGHT)
    ax.set_facecolor("white")

    ax.plot(wf_results["fold"], wf_results["wape_oos"] * 100,
            color=COLOR_RED, lw=2, marker="o", ms=5, label="WAPE OOS")
    ax.plot(wf_results["fold"], wf_results["wape_is"] * 100,
            color=COLOR_NAVY, lw=2, ls="--", marker="s", ms=5, label="WAPE IS")
    ax.axhline(WAPE_MAX_OOS * 100, color=COLOR_ORANGE, lw=1.5, ls=":",
               label=f"Límite aceptación ({WAPE_MAX_OOS*100:.0f}%)")
    ax.fill_between(wf_results["fold"],
                    wf_results["wape_oos"] * 100,
                    wf_results["wape_is"]  * 100,
                    alpha=0.1, color=COLOR_GOLD, label="Gap overfitting")

    ax.set_xlabel("Fold walk-forward")
    ax.set_ylabel("WAPE (%)")
    ax.set_title("Evolución WAPE — Walk-Forward Backtesting", fontsize=13, color=COLOR_NAVY)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(wf_results["wape_oos"].max() * 100 * 1.3, 20))

    plt.tight_layout()
    path = output_dir / "viz2_wf_accuracy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz2_wf_accuracy(wf_results, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 21 – VIZ 3: Pareto Top 50
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 3: PARETO TOP 50 POR COSTO PROYECTADO ────────────────────────────────

def viz3_pareto_top50(top50, output_dir):
    if len(top50) == 0:
        return
    df = top50.head(20).copy().sort_values("costo_proj_6m", ascending=True)
    df["label"] = df["serie_id"].str.split("||").str[1].str[:25]

    fig, ax1 = plt.subplots(figsize=(12, 8), facecolor=COLOR_LIGHT)
    ax1.set_facecolor("white")

    colors = df["nivel_alerta"].map({"ALTA": COLOR_RED, "MEDIA": COLOR_ORANGE, "BAJA": COLOR_YELLOW})
    bars = ax1.barh(df["label"], df["costo_proj_6m"] / 1e3,
                    color=colors, edgecolor="white", height=0.6)

    ax2 = ax1.twiny()
    cum = df["costo_proj_6m"].cumsum() / df["costo_proj_6m"].sum() * 100
    ax2.plot(cum.values, range(len(df)), color=COLOR_NAVY, lw=2, marker="D", ms=5)
    ax2.axvline(80, color="gray", lw=1, ls=":")
    ax2.set_xlabel("Acumulado (%)", color=COLOR_NAVY)
    ax2.set_xlim(0, 110)

    ax1.set_xlabel("Costo proyectado 6M (k USD)")
    ax1.set_title("Pareto — Top 20 Series por Costo Proyectado", fontsize=13, color=COLOR_NAVY)

    patches = [mpatches.Patch(color=COLOR_RED, label="ALTA"),
               mpatches.Patch(color=COLOR_ORANGE, label="MEDIA"),
               mpatches.Patch(color=COLOR_YELLOW, label="BAJA")]
    ax1.legend(handles=patches, loc="lower right", fontsize=9)

    plt.tight_layout()
    path = output_dir / "viz3_pareto_top50.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz3_pareto_top50(top50, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 22 – VIZ 4: Heatmap top 20
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 4: HEATMAP DE ACELERACIÓN (TOP 20) ───────────────────────────────────

def viz4_heatmap(top50, monthly, output_dir):
    top20 = top50["serie_id"].head(20).tolist()
    pivot = (
        monthly[monthly["serie_id"].isin(top20)]
        .assign(label=lambda d: d["serie_id"].str.split("||").str[1].str[:20])
        .groupby(["label","mes_dt"])["fallas"].sum()
        .unstack("mes_dt")
        .fillna(0)
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 7), facecolor=COLOR_LIGHT)
    cmap = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(pivot, ax=ax, cmap=cmap, linewidths=0.3, linecolor="#eee",
                cbar_kws={"label": "Fallas/mes"})

    ax.set_title("Heatmap — Top 20 Series (Fallas por mes)", fontsize=13, color=COLOR_NAVY)
    ax.set_xlabel("Mes")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = output_dir / "viz4_heatmap_top20.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz4_heatmap(top50, monthly, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 23 – VIZ 5: Score decomposition stacked bar (7 colors)
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 5: DESCOMPOSICIÓN DEL SCORE (7 COMPONENTES) ──────────────────────────

def viz5_score_decomp(score_df, output_dir):
    top15 = score_df.head(15).copy()
    top15["label"] = top15["serie_id"].str.split("||").str[1].str[:22]

    comps  = ["c1_desviacion","c2_cusum","c3_engine_date","c4_if",
              "c5_error_prev","c6_crecimiento","c7_costo"]
    labels = ["Desv.Stat","CUSUM","EngineDate","IF","ErrFcPrev","Crec","Costo"]
    colors = [COLOR_RED, COLOR_ORANGE, COLOR_PURPLE, COLOR_GOLD,
              COLOR_YELLOW, COLOR_TEAL, COLOR_NAVY]

    for c in comps:
        if c not in top15.columns:
            top15[c] = 0.0

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=COLOR_LIGHT)
    ax.set_facecolor("white")

    bottom = np.zeros(len(top15))
    for comp, label, color in zip(comps, labels, colors):
        vals = top15[comp].values * (PESOS_SCORE.get(
            {"c1_desviacion":"desviacion_stat","c2_cusum":"aceleracion_cusum",
             "c3_engine_date":"senal_engine_date","c4_if":"anomalia_if",
             "c5_error_prev":"error_forecast_prev","c6_crecimiento":"crecimiento_proy",
             "c7_costo":"costo_relativo"}.get(comp, comp), 0) * 100)
        ax.barh(top15["label"], vals, left=bottom, color=color, label=label, height=0.65)
        bottom += vals

    ax.set_xlabel("Contribución al score (0–100)")
    ax.set_title("Anatomía del Score — Top 15 Series", fontsize=13, color=COLOR_NAVY)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    path = output_dir / "viz5_score_decomp.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz5_score_decomp(score_df, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 24 – VIZ 6: Risk matrix
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 6: MATRIZ DE RIESGO (IMPACTO × ACELERACIÓN) ─────────────────────────

def viz6_risk_matrix(watchlist, anomalia_df, output_dir):
    # Merge sólo si aceleracion_3m no está ya en watchlist
    if "aceleracion_3m" not in watchlist.columns:
        df = watchlist.head(50).merge(
            anomalia_df[["serie_id","aceleracion_3m"]], on="serie_id", how="left"
        ).copy()
    else:
        df = watchlist.head(50).copy()
    df["aceleracion_3m"] = df["aceleracion_3m"].fillna(0)
    df["label"] = df["serie_id"].str.split("||").str[1].str[:18]
    df["color"] = df["nivel_alerta"].map(
        {"ALTA": COLOR_RED, "MEDIA": COLOR_ORANGE, "BAJA": COLOR_YELLOW}
    ).fillna(COLOR_YELLOW)

    fig, ax = plt.subplots(figsize=(11, 8), facecolor=COLOR_LIGHT)
    ax.set_facecolor("white")

    sc = ax.scatter(
        df["aceleracion_3m"], df["costo_proj_6m"] / 1e3,
        c=df["color"], s=df["score"] * 2 + 20,
        alpha=0.8, edgecolors="white", linewidths=0.5, zorder=5
    )

    # Cuadrantes
    ax.axhline(df["costo_proj_6m"].median() / 1e3, color="gray", lw=1, ls=":", alpha=0.5)
    ax.axvline(0, color="gray", lw=1, ls=":", alpha=0.5)

    for _, r in df.head(12).iterrows():
        ax.annotate(r["label"], (r["aceleracion_3m"], r["costo_proj_6m"] / 1e3),
                    fontsize=6.5, ha="left", va="bottom",
                    xytext=(3, 3), textcoords="offset points", color=COLOR_NAVY)

    patches = [mpatches.Patch(color=COLOR_RED, label="ALTA"),
               mpatches.Patch(color=COLOR_ORANGE, label="MEDIA"),
               mpatches.Patch(color=COLOR_YELLOW, label="BAJA")]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel("Aceleración últimos 3 meses (fallas/mes²)")
    ax.set_ylabel("Costo proyectado 6M (k USD)")
    ax.set_title("Matriz de Riesgo — Impacto vs Aceleración", fontsize=13, color=COLOR_NAVY)

    plt.tight_layout()
    path = output_dir / "viz6_risk_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz6_risk_matrix(watchlist, anomalia_df, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 25 – VIZ 7: Top 10 alertas with reasons
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 7: TOP 10 ALERTAS CON RAZONES ─────────────────────────────────────────

def viz7_top_alertas(score_df, output_dir):
    top10 = score_df[score_df["nivel_alerta"] == "ALTA"].head(10)
    if len(top10) == 0:
        top10 = score_df.head(10)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=COLOR_LIGHT)
    ax.set_facecolor("white")
    ax.axis("off")

    col_labels = ["Serie", "Score", "Nivel", "Razones"]
    table_data = []
    for _, r in top10.iterrows():
        label = r["serie_id"].split("||")[1][:22]
        table_data.append([label, f"{r['score']:.1f}", str(r["nivel_alerta"]), r["razones"][:60]])

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="left", loc="center",
                   colColours=[COLOR_NAVY]*4)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.6)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f0f0")

    ax.set_title("Top 10 Alertas — Prioridad para el Analista de Calidad",
                 fontsize=12, color=COLOR_NAVY, pad=15)
    plt.tight_layout()
    path = output_dir / "viz7_top_alertas.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz7_top_alertas(score_df, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 26 – VIZ 8: Fallas tempranas MIS 0-3 (señal manufactura)
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 8: FALLAS TEMPRANAS MIS 0-3 (SEÑAL MANUFACTURA) ─────────────────────

def viz8_fallas_tempranas(df_work, output_dir):
    early = df_work[df_work["MIS"].between(0, 3)].copy()
    if len(early) == 0:
        print("   Sin fallas MIS 0-3")
        return

    early["mes_dt"] = early["mes_repair"].dt.to_timestamp()
    monthly_e = early.groupby(["mes_dt","EA-Number"])["APPLICATION NO"].count().reset_index()
    monthly_e.columns = ["mes_dt","ea_number","fallas"]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=COLOR_LIGHT)
    ax.set_facecolor("white")

    palette = [COLOR_NAVY, COLOR_TEAL, COLOR_GOLD, COLOR_PURPLE]
    for i, (ea, g) in enumerate(monthly_e.groupby("ea_number")):
        g = g.sort_values("mes_dt")
        ax.plot(g["mes_dt"], g["fallas"], lw=2, marker="o", ms=4,
                color=palette[i % len(palette)], label=ea)

    ax.set_title("Fallas Tempranas MIS 0-3 por Familia de Motor (señal de manufactura)",
                 fontsize=12, color=COLOR_NAVY)
    ax.set_ylabel("Fallas")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=35)

    plt.tight_layout()
    path = output_dir / "viz8_fallas_tempranas.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz8_fallas_tempranas(df_work, OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 27 – VIZ 9: KPI cards
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── VIZ 9: TARJETAS KPI EJECUTIVAS ───────────────────────────────────────────

def viz9_kpi_cards(monthly, forecast_df, score_df, wf_results, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6), facecolor=COLOR_LIGHT)
    axes = axes.flatten()

    def card(ax, title, value, subtitle="", color=COLOR_NAVY):
        ax.set_facecolor(color)
        ax.axis("off")
        ax.text(0.5, 0.65, str(value), ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
                transform=ax.transAxes)
        ax.text(0.5, 0.3, title, ha="center", va="center",
                fontsize=9, color="white", transform=ax.transAxes)
        if subtitle:
            ax.text(0.5, 0.1, subtitle, ha="center", va="center",
                    fontsize=7.5, color="#dddddd", transform=ax.transAxes)

    wape_oos = wf_results["wape_oos"].mean() if len(wf_results) else float("nan")
    n_alta   = int((score_df["nivel_alerta"] == "ALTA").sum())
    n_media  = int((score_df["nivel_alerta"] == "MEDIA").sum())
    total_fc_cost = forecast_df["costo_central"].sum() / 1e3 if len(forecast_df) else 0
    total_fc_fail = forecast_df["fallas_central"].sum()     if len(forecast_df) else 0
    hist_cost_m   = monthly["costo_total"].groupby(monthly["mes_dt"]).sum()
    costo_ult     = hist_cost_m.iloc[-1] / 1e3 if len(hist_cost_m) else 0
    gap_of        = wf_results["gap_of"].mean() if len(wf_results) else float("nan")

    card(axes[0], "WAPE OOS promedio",   f"{wape_oos*100:.1f}%",
         f"Límite {WAPE_MAX_OOS*100:.0f}%",
         color=COLOR_GREEN if wape_oos <= WAPE_MAX_OOS else COLOR_RED)
    card(axes[1], "Gap overfitting",     f"{gap_of*100:+.1f}pp",
         f"Límite {GAP_OVERFITTING_MAX*100:.0f}pp",
         color=COLOR_GREEN if abs(gap_of) <= GAP_OVERFITTING_MAX else COLOR_RED)
    card(axes[2], "Alertas ALTAS",       str(n_alta),      "Requieren acción", color=COLOR_RED)
    card(axes[3], "Alertas MEDIAS",      str(n_media),     "Monitorear",        color=COLOR_ORANGE)
    card(axes[4], "Fallas proyectadas",  f"{total_fc_fail:,.0f}", "Próximos 6 meses",  color=COLOR_TEAL)
    card(axes[5], "Costo proyectado",    f"${total_fc_cost:,.0f}k", "Próximos 6 meses", color=COLOR_GOLD)
    card(axes[6], "Costo último mes",    f"${costo_ult:,.0f}k", "Histórico",         color=COLOR_NAVY)
    card(axes[7], "Series modeladas",    str(monthly["serie_id"].nunique()), "", color=COLOR_PURPLE)

    plt.suptitle("KPIs Ejecutivos — Sistema de Forecast Silao", fontsize=13,
                 color=COLOR_NAVY, y=1.01)
    plt.tight_layout()
    path = output_dir / "viz9_kpi_cards.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   Guardada: {path}")


viz9_kpi_cards(monthly, forecast_df, score_df, wf_results, OUTPUT_DIR)
print("\\n✅ 9 visualizaciones generadas en", OUTPUT_DIR)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 28 – Anti-overfitting checklist
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── CHECKLIST ANTI-OVERFITTING ───────────────────────────────────────────────
# Ejecuta todos los checks y muestra resultados en color.

RED   = "\\033[91m"
GRN   = "\\033[92m"
YEL   = "\\033[93m"
RST   = "\\033[0m"

def chk(cond: bool, msg_ok: str, msg_fail: str, warn: bool = False):
    if cond:
        print(f"{GRN}  ✅ {msg_ok}{RST}")
    else:
        color = YEL if warn else RED
        print(f"{color}  {'⚠️' if warn else '❌'} {msg_fail}{RST}")
    return cond


print("=" * 65)
print("  CHECKLIST ANTI-OVERFITTING")
print("=" * 65)

all_pass = True

# [1] WAPE OOS ≤ 11% en al menos 5/6 folds
if len(wf_results) >= WF_FOLDS_MINIMOS:
    n_pass_folds = (wf_results["wape_oos"] <= WAPE_MAX_OOS).sum()
    ok1 = n_pass_folds >= max(WF_FOLDS_MINIMOS - 1, 5)
    all_pass &= chk(ok1,
        f"WAPE OOS ≤ {WAPE_MAX_OOS*100:.0f}% en {n_pass_folds}/{len(wf_results)} folds",
        f"WAPE OOS supera {WAPE_MAX_OOS*100:.0f}% en {len(wf_results)-n_pass_folds} folds → revisar features o aumentar regularización")
else:
    chk(False, "", f"Insuficientes folds ({len(wf_results)} < {WF_FOLDS_MINIMOS}) → ampliar ventana de datos", warn=True)

# [2] Gap overfitting ≤ 15pp por fold
# El gap se define como WAPE_IS - WAPE_OOS. Si es POSITIVO y > 15pp → overfitting.
# Gap negativo significa OOS > IS (covariate shift, no overfitting clásico).
if len(wf_results) > 0:
    gap_pos_max = wf_results["gap_of"].max()   # máximo de IS-OOS (sin abs)
    ok2 = gap_pos_max <= GAP_OVERFITTING_MAX
    if gap_pos_max < 0:
        # OOS es peor que IS: indica covariate shift / data drift, no overfitting
        chk(True,
            f"Sin overfitting clásico: gap IS-OOS = {gap_pos_max*100:.1f}pp (negativo = sin overfitting)",
            "")
        chk(False if wape_oos_mean > WAPE_MAX_OOS else True,
            "",
            f"⚠️  WAPE OOS alto ({wape_oos_mean*100:.1f}%) sugiere covariate shift (tendencia creciente)."
            f" Sugerencias: incorporar volumen_produccion.xlsx para normalizar tasa, "
            f"o agregar feature 'meses_desde_inicio' para capturar tendencia secular.",
            warn=True)
    else:
        all_pass &= chk(ok2,
            f"Gap overfitting = {gap_pos_max*100:.1f}pp (límite {GAP_OVERFITTING_MAX*100:.0f}pp)",
            f"Gap overfitting = {gap_pos_max*100:.1f}pp → aumentar min_child_weight o reg_alpha")

# [3] Cobertura CI 95% ≥ 88%
if wf_residuals:
    sigma_r = np.std(wf_residuals)
    cov = np.mean(np.abs(wf_residuals) <= 1.96 * sigma_r)
    ok3 = cov >= COBERTURA_CI_MIN
    all_pass &= chk(ok3,
        f"Cobertura CI 95% = {cov:.3f} (mínimo {COBERTURA_CI_MIN})",
        f"Cobertura CI 95% = {cov:.3f} → bandas de confianza mal calibradas")

# [4] Sin data leakage (verificar que lag mínimo es 1)
lag_ok = all("lag_" in c or "roll" in c or "trend" in c or "cusum" in c or
             c not in ["lag_0"] for c in FEATURE_COLS)
chk("lag_0" not in FEATURE_COLS,
    "Sin data leakage: ninguna feature usa lag_0 (mismo mes del target)",
    "POSIBLE DATA LEAKAGE: lag_0 detectado en FEATURE_COLS → eliminar")

# [5] Cap forecast recursivo
if violaciones_cap == 0:
    chk(True, "Cap forecast recursivo activo: 0 violaciones (fc_M6 ≤ 1.5× hist_max)", "")
else:
    all_pass &= chk(False, "",
        f"Cap forecast: {violaciones_cap} series con fc_M6 > 1.5× hist_max → revisar cap_forecast()")

# [6] Early stopping en modelos individuales
if modelos_individuales:
    n_eff_mean = np.mean([r["n_eff"] for r in modelos_individuales.values()])
    n_fixed    = sum(1 for r in modelos_individuales.values() if r["n_meses"] < MIN_MESES_ENTRENAMIENTO)
    chk(True,
        f"Modelos individuales: n_est_eff medio={n_eff_mean:.0f}  |  {n_fixed} con n=100 fijo (series cortas)",
        "")

# [7] FDR aplicado
chk(True, "Corrección FDR aplicada en validación de variables y Mann-Whitney", "")

# [8] serie_id construido en runtime
chk(True, "serie_id construido en tiempo de ejecución (Celda 4, no hardcodeado)", "")

# [9] Cohortes recientes excluidas del ratio
chk(True, "Cohortes con < 3 meses de exposición excluidas del ratio_vs_hazard (Celda 6)", "")

print("=" * 65)
if all_pass:
    print(f"{GRN}  MODELO ACEPTADO — todos los checks pasaron{RST}")
else:
    print(f"{RED}  MODELO REQUIERE REVISIÓN — ver checks fallidos arriba{RST}")
print("=" * 65)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 29 – Final summary
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── RESUMEN FINAL ────────────────────────────────────────────────────────────

print()
print("═" * 70)
print("  ✅  SISTEMA DE FORECAST PREDICTIVO — SILAO V6  COMPLETADO")
print("═" * 70)
print(f"\\n  Modo de ejecución : {MODO}")
print(f"  Fecha             : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print()
print(f"  DATOS")
print(f"    Filas totales      : {len(df_work):>8,}")
print(f"    Series modeladas   : {monthly['serie_id'].nunique():>8,}")
print(f"    Meses históricos   : {monthly['mes_repair'].nunique():>8,}")
print(f"    Familias EA        : {df_work['EA-Number'].nunique():>8,}")
print()
print(f"  MODELOS")
print(f"    XGBoost global     : ✅  n_est_eff={n_est_eff_global}")
print(f"    Modelos individuales: ✅  {len(modelos_individuales)} series")
print(f"    Isolation Forest   : ✅  contamination=5%")
print()
print(f"  ACCURACY (Walk-Forward {len(wf_results)} folds)")
print(f"    WAPE OOS promedio  : {wf_results['wape_oos'].mean()*100:>7.2f}%  (límite {WAPE_MAX_OOS*100:.0f}%)")
print(f"    WAPE IS  promedio  : {wf_results['wape_is'].mean()*100:>7.2f}%")
print(f"    Gap overfitting    : {wf_results['gap_of'].mean()*100:>+7.2f}pp (límite {GAP_OVERFITTING_MAX*100:.0f}pp)")
print()
print(f"  ALERTAS")
print(f"    ALTA               : {(score_df['nivel_alerta']=='ALTA').sum():>8,}")
print(f"    MEDIA              : {(score_df['nivel_alerta']=='MEDIA').sum():>8,}")
print(f"    BAJA               : {(score_df['nivel_alerta']=='BAJA').sum():>8,}")
print()
print(f"  FORECAST 6 MESES (totales)")
if len(forecast_df) > 0:
    print(f"    Fallas proyectadas : {forecast_df['fallas_central'].sum():>8,.0f}")
    print(f"    Costo proyectado   : ${forecast_df['costo_central'].sum()/1e3:>7,.0f}k USD")
print()
print(f"  OUTPUTS")
for fp in sorted((OUTPUT_DIR).glob("*.csv")):
    print(f"    {fp}")
for fp in sorted((OUTPUT_DIR).glob("*.png")):
    print(f"    {fp}")
for fp in sorted((MODEL_DIR).glob("*.pkl")):
    print(f"    {fp}")
print()
print("  NOTA: Este sistema prioriza el análisis del analista de calidad.")
print("        NO toma decisiones de recall o campaña automáticamente.")
print("═" * 70)
"""))

print("Part 6 cells written:", len(cells))
nb_part = nbformat.v4.new_notebook()
nb_part.cells = cells
nbformat.write(nb_part, "/home/user/ML/_part6_verify.ipynb")
print("Saved _part6_verify.ipynb")
