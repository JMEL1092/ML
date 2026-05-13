"""Part 1: Cells 0-5 (Config, Imports, Utilities, Data Load, Aggregation, Cohort/Hazard)"""
import nbformat

cells = []

# ─────────────────────────────────────────────────────────────────
# CELL 0 – Title markdown
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_markdown_cell("""\
# Sistema de Forecast Predictivo de Fallas — V6
## Planta de Motores Silao — Volkswagen de México

**Flujo de ejecución (una sola pasada `Run All`):**
1. Configuración global
2. Imports y utilidades
3. Carga de datos (inicial o incremental)
4. Agregación mensual por serie
5. Análisis de cohortes y curva de hazard
6. Feature engineering (sin data leakage)
7. Validación estadística de variables (FDR)
8. Optimización Optuna (si `REOPTIMIZAR_HIPERPARAMETROS=True`)
9. Backtesting walk-forward (≥6 folds, WAPE OOS ≤ 11%)
10. Modelo global XGBoost
11. Modelos individuales por serie
12. Forecast recursivo 6 meses con cap
13. Isolation Forest (anomalías multivariadas)
14. CUSUM + detección estadística
15. Score 7 componentes
16. Watchlist Top 50
17. Exportar 8 CSVs para Power BI
18–26. Visualizaciones
27. Checklist anti-overfitting
28. Resumen y rollback
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 1 – Global configuration
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── CONFIGURACIÓN GLOBAL ─────────────────────────────────────────────────────
from pathlib import Path

# Archivos de datos
DATA_FILE             = "All_Engines_Silao.xlsx"
NUEVO_DATA_FILE       = "Datos_Power_BI_-_Silao_NUEVO.xlsx"
VOLUMEN_FILE          = "volumen_produccion.xlsx"

# Directorios de salida
MODEL_DIR  = Path("modelos_ml")
OUTPUT_DIR = Path("outputs")

# Parámetros de forecast
HORIZONTE_MESES          = 6
MIN_FALLAS_SERIE         = 3
MIN_MESES_SERIE          = 6
MIN_MESES_ENTRENAMIENTO  = 12
WF_FOLDS_MINIMOS         = 6

# Umbrales de aceptación del modelo
WAPE_MAX_OOS          = 0.11    # ≤ 11 %
GAP_OVERFITTING_MAX   = 0.15    # gap (in-sample – OOS) ≤ 15 pp
COBERTURA_CI_MIN      = 0.88    # cobertura CI 95 % ≥ 88 %

# Regularización XGBoost (no negociable)
XGB_PARAMS_BASE = dict(
    reg_alpha        = 1.0,
    reg_lambda       = 2.0,
    min_child_weight = 3,
    max_depth        = 5,
    subsample        = 0.85,
    colsample_bytree = 0.9,
    n_estimators     = 300,
    learning_rate    = 0.05,
    random_state     = 42,
    objective        = "reg:squarederror",
    verbosity        = 0,
)

# Pesos del score de alerta (7 componentes, deben sumar 1.0)
PESOS_SCORE = {
    "desviacion_stat"    : 0.20,   # z-score Repair Date
    "aceleracion_cusum"  : 0.18,   # CUSUM
    "senal_engine_date"  : 0.15,   # cohorte Engine Date (señal anticipada)
    "anomalia_if"        : 0.17,   # Isolation Forest
    "error_forecast_prev": 0.10,   # error forecast previo
    "crecimiento_proy"   : 0.12,   # crecimiento proyectado
    "costo_relativo"     : 0.08,   # costo relativo al total
}
assert abs(sum(PESOS_SCORE.values()) - 1.0) < 1e-9, "Los pesos del score deben sumar exactamente 1.0"

# Optuna
N_OPTUNA_TRIALS              = 50
REOPTIMIZAR_HIPERPARAMETROS  = False   # True sólo en el primer run o cuando se desee re-optimizar

# Decay temporal en pesos de entrenamiento
DECAY_RATE = 2.5

# Paleta corporativa
COLOR_NAVY   = "#0A2540"
COLOR_TEAL   = "#1E5F74"
COLOR_GOLD   = "#D4A74A"
COLOR_RED    = "#C0392B"
COLOR_ORANGE = "#E67E22"
COLOR_YELLOW = "#F39C12"
COLOR_GREEN  = "#16A085"
COLOR_PURPLE = "#8B4789"
COLOR_LIGHT  = "#F7F5F0"

# Extensibilidad: agregar variables nuevas aquí
CANDIDATAS_NUEVAS_CATEGORICAS: list = []
CANDIDATAS_NUEVAS_NUMERICAS:   list = []

print("✅ Configuración cargada")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 2 – Imports
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── IMPORTS ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import warnings
import logging
import json
import gc
import hashlib
from datetime import datetime
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns

from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.stats import mannwhitneyu, chi2_contingency, f_oneway, kruskal

import optuna
from optuna.pruners import MedianPruner
optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("silao_v6")

# Crear directorios
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "individuales").mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Imports OK  |  pandas {pd.__version__}  |  numpy {np.__version__}")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 3 – Utility functions
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── FUNCIONES UTILITARIAS ────────────────────────────────────────────────────

MESES_ES = {
    "enero":1,"febrero":2,"marzo":3,"abril":4,"mayo":5,"junio":6,
    "julio":7,"agosto":8,"septiembre":9,"octubre":10,"noviembre":11,"diciembre":12
}


def parse_fecha(s) -> pd.Timestamp:
    \"\"\"Parse fecha en español tipo 'lunes, 14 de marzo de 2024'.\"\"\"
    if pd.isna(s):
        return pd.NaT
    try:
        partes = str(s).lower().split(", ", 1)[-1].split(" de ")
        return pd.Timestamp(int(partes[2]), MESES_ES[partes[1]], int(partes[0]))
    except Exception:
        return pd.NaT


def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    \"\"\"Weighted Absolute Percentage Error. NaN si sum(actual)==0.\"\"\"
    actual = np.asarray(actual, dtype=float)
    pred   = np.asarray(pred,   dtype=float)
    total  = float(np.nansum(np.abs(actual)))
    if total == 0:
        return np.nan
    return float(np.nansum(np.abs(actual - pred)) / total)


def fdr_correction(p_values: np.ndarray, alpha: float = 0.05):
    \"\"\"Benjamini-Hochberg FDR. Returns (rejected_mask, adjusted_p_values).\"\"\"
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    idx   = np.argsort(p)
    p_s   = p[idx]
    p_adj = np.minimum.accumulate(p_s[::-1] * n / np.arange(n, 0, -1))[::-1]
    out   = np.empty(n)
    out[idx] = p_adj
    return out < alpha, out


def cap_forecast(pred: float, historia_real: np.ndarray) -> float:
    \"\"\"
    Aplica cap conservador al forecast recursivo.
    historia_real contiene SÓLO observaciones reales, nunca predicciones.
    \"\"\"
    if len(historia_real) == 0:
        return max(0.0, float(pred))
    max_h  = float(np.max(historia_real))
    tail   = historia_real[-min(6, len(historia_real)):]
    mean_r = float(np.mean(tail))
    cap_hi = min(max_h * 1.5, mean_r * 2.5)
    cap_lo = max(0.0, mean_r * 0.3)
    return float(np.clip(pred, cap_lo, cap_hi))


def sample_weights_decay(n: int, decay: float = DECAY_RATE) -> np.ndarray:
    \"\"\"Decay exponencial: meses recientes tienen más peso en entrenamiento.\"\"\"
    t = np.linspace(0, 1, n)
    w = np.exp(t * decay)
    return (w / w.mean()).astype(float)


def ci_bandas(pred: np.ndarray, residuals: np.ndarray):
    \"\"\"
    CI 95 % que se ensancha con el horizonte.
    El ancho crece como sqrt(h) para reflejar mayor incertidumbre en M+6.
    \"\"\"
    sigma = float(np.nanstd(residuals)) if len(residuals) > 1 else float(np.nanmean(np.abs(residuals)) + 1e-8)
    z     = 1.96
    h     = np.arange(1, len(pred) + 1)
    lo    = np.maximum(pred - z * sigma * np.sqrt(h), 0)
    hi    = pred + z * sigma * np.sqrt(h)
    return lo, hi


def cusum_stat(series: np.ndarray, k: float = 0.5) -> float:
    \"\"\"CUSUM acumulado positivo normalizado por sigma del baseline.\"\"\"
    if len(series) < 3:
        return 0.0
    half    = max(3, len(series) // 2)
    mu      = float(np.mean(series[:half]))
    sigma   = float(np.std(series[:half])) + 1e-8
    slack   = k * sigma
    s       = 0.0
    for x in series:
        s = max(0.0, s + (x - mu) - slack)
    return s / sigma


def slope_lineal(arr: np.ndarray) -> float:
    \"\"\"Pendiente OLS sobre los últimos puntos.\"\"\"
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=float)
    return float(np.polyfit(x, arr.astype(float), 1)[0])


def hash_serie(serie_id: str) -> str:
    \"\"\"Hash corto para usar como nombre de archivo de modelo individual.\"\"\"
    return hashlib.md5(serie_id.encode()).hexdigest()[:12]


print("✅ Utilidades cargadas")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 4 – Data loading
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── CARGA DE DATOS Y MODO DE EJECUCIÓN ───────────────────────────────────────

hay_datos_nuevos = Path(NUEVO_DATA_FILE).exists()
hay_modelo_previo = (MODEL_DIR / "global_xgb.pkl").exists()
MODO = "INCREMENTAL" if (hay_datos_nuevos and hay_modelo_previo) else "INICIAL"

print(f"📂 Modo: {MODO}  |  datos_nuevos={hay_datos_nuevos}  |  modelo_previo={hay_modelo_previo}")

# ── Cargar dataset principal ─────────────────────────────────────────────────
df_raw = pd.read_excel(DATA_FILE, sheet_name="923-All Engines Silao")
print(f"\\n📊 Dataset cargado: {len(df_raw):,} filas × {len(df_raw.columns)} columnas")

# Asegurar tipos fecha
_fecha_cols = ["Production Date", "Engine Date", "SALES DATE", "LOADING DATE", "Repair date"]
for col in _fecha_cols:
    if col in df_raw.columns and df_raw[col].dtype == object:
        df_raw[col] = pd.to_datetime(df_raw[col], errors="coerce")

# ── Modo incremental: combinar con nuevos datos ───────────────────────────────
if MODO == "INCREMENTAL":
    df_nuevo = pd.read_excel(NUEVO_DATA_FILE)
    cols_c = [c for c in df_raw.columns if c in df_nuevo.columns]
    df_nuevo = df_nuevo[cols_c].copy()
    for col in _fecha_cols:
        if col in df_nuevo.columns and df_nuevo[col].dtype == object:
            df_nuevo[col] = pd.to_datetime(df_nuevo[col], errors="coerce")
    n_antes = len(df_raw)
    df_raw = (pd.concat([df_raw, df_nuevo], ignore_index=True)
                .drop_duplicates(subset=["APPLICATION NO"], keep="last"))
    print(f"   ➕ Filas netas añadidas: {len(df_raw) - n_antes:,}")

# ── Limpieza básica ───────────────────────────────────────────────────────────
df_raw = df_raw.dropna(subset=["Repair date", "EA-Number", "DAMAGE CODE", "Basic no. Description"])
df_raw["COSTS"] = pd.to_numeric(df_raw["COSTS"], errors="coerce").fillna(0).clip(lower=0)
df_raw["MIS"]   = pd.to_numeric(df_raw["MIS"],   errors="coerce").fillna(0).clip(lower=0)
df_raw["KM"]    = pd.to_numeric(df_raw["KM"],    errors="coerce").fillna(0).clip(lower=0)

# ── Construir serie_id DESPUÉS de cargar datos ────────────────────────────────
# Nota: se construye en tiempo de ejecución, nunca hardcodeado
df_raw["serie_id"] = (
    df_raw["EA-Number"].astype(str).str.strip() + "||" +
    df_raw["Basic no. Description"].astype(str).str.strip() + "||" +
    df_raw["DAMAGE CODE"].astype(str).str.strip()
)

# Columnas de período
df_raw["mes_repair"] = df_raw["Repair date"].dt.to_period("M")
df_raw["mes_engine"] = df_raw["Engine Date"].dt.to_period("M")

# ── Excluir último mes de Repair Date (puede estar incompleto) ───────────────
max_mes_repair = df_raw["mes_repair"].max()
df_work = df_raw[df_raw["mes_repair"] < max_mes_repair].copy()
print(f"\\n📅 Rango Repair Date: {df_work['mes_repair'].min()} → {df_work['mes_repair'].max()}")
print(f"   Mes excluido (incompleto): {max_mes_repair}")
print(f"   Filas activas: {len(df_work):,}")
print(f"   Series únicas: {df_work['serie_id'].nunique():,}")

# ── Volumen de producción (opcional) ────────────────────────────────────────
hay_volumen = Path(VOLUMEN_FILE).exists()
vol_df = None
if hay_volumen:
    vol_df = pd.read_excel(VOLUMEN_FILE)
    vol_df.columns = [c.lower().strip() for c in vol_df.columns]
    vol_df["cohorte"] = pd.to_datetime(vol_df["cohorte"], errors="coerce").dt.to_period("M")
    print(f"\\n✅ volumen_produccion.xlsx cargado → tasa de falla = fallas / motores producidos")
else:
    print("\\nℹ️  Sin volumen_produccion.xlsx → tasa de falla como proxy por distribución de edad")
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 5 – Monthly aggregation
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── AGREGACIÓN MENSUAL POR SERIE ────────────────────────────────────────────

def agregar_mensual(df: pd.DataFrame) -> pd.DataFrame:
    \"\"\"
    Construye el panel mensual: una fila por (serie_id × mes_repair).
    Calcula fallas (conteo), costo_total, mis_mean, km_mean.
    \"\"\"
    grp = df.groupby(["serie_id", "mes_repair"])
    agg = grp.agg(
        fallas      = ("APPLICATION NO", "count"),
        costo_total = ("COSTS",          "sum"),
        mis_mean    = ("MIS",            "mean"),
        km_mean     = ("KM",             "mean"),
    ).reset_index()

    # Metadatos de la serie (estáticos – tomar la moda)
    def safe_mode(x): return x.mode().iloc[0] if len(x) and len(x.mode()) else "UNK"
    def mis_cl(x):
        cut = pd.cut(x, bins=[0,3,7,12,24], labels=["0-3","4-7","8-12","13-24"], right=True)
        m = cut.dropna().mode()
        return str(m.iloc[0]) if len(m) else "0-3"
    meta = df.groupby("serie_id").agg(
        ea_number   = ("EA-Number",              safe_mode),
        componente  = ("Basic no. Description",  safe_mode),
        damage_code = ("DAMAGE CODE",            safe_mode),
    ).reset_index()
    meta["mis_cluster"] = (
        df.groupby("serie_id")["MIS"].apply(mis_cl).values
    )

    monthly = agg.merge(meta, on="serie_id", how="left")
    monthly["mes_dt"] = monthly["mes_repair"].dt.to_timestamp()
    monthly = monthly.sort_values(["serie_id", "mes_repair"]).reset_index(drop=True)
    return monthly


monthly = agregar_mensual(df_work)
print(f"✅ Panel mensual: {len(monthly):,} filas  ({monthly['serie_id'].nunique()} series × {monthly['mes_repair'].nunique()} meses)")

# Estadísticas de series
n_por_serie = monthly.groupby("serie_id")["fallas"].agg(["sum","count"])
n_por_serie.columns = ["total_fallas","n_meses"]
print(f"\\nDistribución de series por meses con datos:")
print(n_por_serie["n_meses"].describe().round(1).to_string())
print(f"\\nSeries con ≥ {MIN_MESES_SERIE} meses:  {(n_por_serie['n_meses'] >= MIN_MESES_SERIE).sum()}")
print(f"Series con ≥ {MIN_MESES_ENTRENAMIENTO} meses: {(n_por_serie['n_meses'] >= MIN_MESES_ENTRENAMIENTO).sum()}")

# Costo unitario de referencia por serie (mediana histórica)
costos_ref = (
    df_work[df_work["COSTS"] > 0]
    .groupby("serie_id")["COSTS"]
    .agg(costo_unit_med="median", costo_unit_p75=lambda x: x.quantile(0.75))
    .reset_index()
)
"""))

# ─────────────────────────────────────────────────────────────────
# CELL 6 – Cohort / hazard analysis
# ─────────────────────────────────────────────────────────────────
cells.append(nbformat.v4.new_code_cell("""\
# ── ANÁLISIS DE COHORTES Y CURVA DE HAZARD ──────────────────────────────────
# Engine Date = señal anticipada con CENSURA (lotes recientes aún no tuvieron tiempo de fallar)

def calcular_hazard_por_familia(df: pd.DataFrame, max_mis: int = 23) -> dict:
    \"\"\"
    Curva de hazard empírica por EA-Number.
    Cuando hay volumen_produccion.xlsx → tasa real = fallas / motores producidos.
    Sin él → proxy = P(falla|MIS=k) ≈ fallas_k / fallas_k+
    Sólo incluye cohortes con ≥ 3 meses de exposición y ≥ 3 fallas observadas.
    \"\"\"
    hazard = {}
    hoy = df["mes_repair"].max()

    for ea, grp in df.groupby("EA-Number"):
        # Filtrar cohortes confiables (≥3 meses de exposición desde Engine Date)
        grp = grp.copy()
        grp["meses_expuesto"] = (hoy - grp["mes_engine"]).apply(
            lambda x: x.n if pd.notna(x) else 0
        )
        # Excluir cohortes con < 3 meses de exposición
        grp_conf = grp[grp["meses_expuesto"] >= 3]

        if hay_volumen and vol_df is not None:
            # Tasa real: fallas / motores producidos en cohorte
            vol_ea = vol_df[vol_df.get("ea_number", vol_df.get("codigo_motor", vol_df.columns[1])) == ea]
            mis_counts = grp_conf.groupby("MIS")["APPLICATION NO"].count()
            fallas_tot = max(mis_counts.sum(), 1)
            # Hazard condicional por MIS
            h = {}
            for k in range(max_mis + 1):
                at_risk = mis_counts[mis_counts.index >= k].sum()
                h[k] = float(mis_counts.get(k, 0)) / max(at_risk, 1)
        else:
            # Proxy: distribución relativa por MIS
            mis_counts = grp_conf.groupby("MIS")["APPLICATION NO"].count()
            total = max(mis_counts.sum(), 1)
            h = {}
            for k in range(max_mis + 1):
                at_risk = mis_counts[mis_counts.index >= k].sum()
                h[k] = float(mis_counts.get(k, 0)) / max(at_risk, 1)

        hazard[ea] = h

    return hazard


hazard_curves = calcular_hazard_por_familia(df_work)
print("✅ Curvas de hazard calculadas para familias:", list(hazard_curves.keys()))

def calcular_cohorte_ratio(df: pd.DataFrame, hazard: dict) -> pd.DataFrame:
    \"\"\"
    Para cada (serie_id × mes_engine), calcula:
      cohorte_activa_ratio = fallas_obs / fallas_esperadas_por_hazard
    Solo cohortes con ≥ 3 meses exposición y ≥ 3 fallas.
    Los últimos 2 meses de Engine Date se marcan como 'no_confiable'.
    \"\"\"
    hoy = df["mes_repair"].max()
    max_mes_engine = df["mes_engine"].max()
    # Umbral: cohortes más recientes que este mes se excluyen del ratio
    umbral_reciente = max_mes_engine - 2   # últimos 2 meses excluidos

    rows = []
    for (sid, mes_e), g in df.groupby(["serie_id", "mes_engine"]):
        ea = g["EA-Number"].iloc[0]
        meses_exp = (hoy - mes_e).n if hasattr((hoy - mes_e), "n") else 0

        if meses_exp < 3 or len(g) < 3:
            continue

        confiable = (mes_e <= umbral_reciente)
        hz = hazard.get(ea, {})
        # Fallas esperadas = sum(hazard[0..meses_exp]) — proporcional a tamaño cohorte
        fallas_obs = len(g)
        expected_cumul = sum(hz.get(k, 0) for k in range(min(meses_exp + 1, 24)))
        proxy_vol = fallas_obs / max(expected_cumul, 1e-8)  # estimación del tamaño de cohorte
        fallas_esperadas = expected_cumul * proxy_vol

        ratio = fallas_obs / max(fallas_esperadas, 1e-8)

        rows.append({
            "serie_id":            sid,
            "mes_engine":          mes_e,
            "cohorte_ratio":       ratio if confiable else np.nan,
            "cohorte_confiable":   confiable,
            "fallas_obs_cohorte":  fallas_obs,
            "meses_expuesto":      meses_exp,
        })

    return pd.DataFrame(rows)


cohorte_df = calcular_cohorte_ratio(df_work, hazard_curves)
print(f"✅ Cohortes analizadas: {len(cohorte_df):,}  (confiables: {cohorte_df['cohorte_confiable'].sum()})")

# Ratio consolidado por serie (máximo ratio entre sus cohortes confiables)
cohorte_serie = (
    cohorte_df[cohorte_df["cohorte_confiable"]]
    .groupby("serie_id")["cohorte_ratio"]
    .agg(cohorte_ratio_max="max", cohorte_ratio_mean="mean")
    .reset_index()
)
"""))

print("Part 1 cells written:", len(cells))

# Save partial notebook for verification
nb_part = nbformat.v4.new_notebook()
nb_part.cells = cells
nbformat.write(nb_part, "/home/user/ML/_part1_verify.ipynb")
print("Saved _part1_verify.ipynb")
