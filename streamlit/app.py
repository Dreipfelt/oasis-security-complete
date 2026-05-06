"""
OASIS Security – Tableau de bord de la délinquance en France
Données : data.gouv.fr – Bases statistiques communale/départementale/régionale
Déployable sur Hugging Face Spaces (Streamlit)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import io
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OASIS Security – Délinquance France",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_URL = (
    "https://static.data.gouv.fr/resources/"
    "bases-statistiques-communale-departementale-et-regionale-de-la-delinquance-"
    "enregistree-par-la-police-et-la-gendarmerie-nationales/"
    "20260129-160256/donnee-reg-data.gouv-2025-geographie2025-produit-le2026-01-22.csv"
)

GEOJSON_URL = (
    "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements-version-simplifiee.geojson"
)

# ─────────────────────────────────────────────
# CUSTOM CSS – couleurs sombres inspirées d'OASIS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Fond général */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    /* Cartes métriques */
    div[data-testid="metric-container"] {
        background-color: #1c2128;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; }
    div[data-testid="metric-container"] div { color: #e6edf3 !important; }
    /* Titres */
    h1, h2, h3 { color: #58a6ff; }
    /* Séparateurs */
    hr { border-color: #30363d; }
    /* Multiselect tags */
    .stMultiSelect span[data-baseweb="tag"] { background-color: #1f6feb; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="📥 Chargement du dataset data.gouv.fr…")
def load_data():
    try:
        df = pd.read_csv(DATASET_URL, sep=";", encoding="utf-8", low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(DATASET_URL, sep=",", encoding="utf-8", low_memory=False)
        except Exception as e:
            st.error(f"Impossible de charger le dataset : {e}")
            return None
    return df


@st.cache_data(show_spinner="🗺️ Chargement du GeoJSON…")
def load_geojson():
    try:
        r = requests.get(GEOJSON_URL, timeout=15)
        return r.json()
    except Exception as e:
        st.warning(f"GeoJSON non disponible : {e}")
        return None


# ─────────────────────────────────────────────
# DÉTECTION AUTOMATIQUE DES COLONNES
# ─────────────────────────────────────────────
def detect_columns(df: pd.DataFrame):
    """
    Détecte automatiquement les colonnes clés quel que soit le schéma exact du CSV.
    Retourne un dict: {role: nom_colonne}
    """
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    # Colonne département
    for candidate in ["num_dep", "dep", "codgeo", "code_dep", "departement", "code_departement", "numdep"]:
        if candidate in cols:
            mapping["dep"] = cols[candidate]
            break

    # Colonne libellé département
    for candidate in ["lib_dep", "libgeo", "nom_dep", "departement", "libelle_departement", "libelle"]:
        if candidate in cols and candidate != mapping.get("dep", "").lower():
            mapping["lib_dep"] = cols[candidate]
            break

    # Colonne année
    for candidate in ["annee", "year", "an", "annee_statistique"]:
        if candidate in cols:
            mapping["annee"] = cols[candidate]
            break

    # Colonne type crime/délit
    for candidate in ["classe", "indicateur", "type_crime", "libelle_index", "index", "faits", "libelle_classe", "crime"]:
        if candidate in cols:
            mapping["classe"] = cols[candidate]
            break

    # Colonne valeur / nombre de faits
    for candidate in ["faits", "nb_faits", "valeur", "nombre", "count", "total", "nbr_faits", "nombre_faits"]:
        if candidate in cols and candidate != mapping.get("classe", "").lower():
            mapping["valeur"] = cols[candidate]
            break

    # Colonne taux
    for candidate in ["taux", "taux_pour_mille", "tx", "taux_faits", "taux_criminalite"]:
        if candidate in cols:
            mapping["taux"] = cols[candidate]
            break

    # Colonne population
    for candidate in ["pop", "population", "pop_legale", "population_municipale"]:
        if candidate in cols:
            mapping["pop"] = cols[candidate]
            break

    return mapping


# ─────────────────────────────────────────────
# PRÉVISIONS
# ─────────────────────────────────────────────
def forecast_series(series: pd.Series, horizon: int = 5) -> pd.Series:
    """
    Applique Holt-Winters ou une régression linéaire simple pour prévoir
    les valeurs jusqu'à 2030.
    """
    series = series.dropna()
    if len(series) < 3:
        return pd.Series(dtype=float)

    last_year = int(series.index[-1])
    future_years = list(range(last_year + 1, last_year + horizon + 1))

    try:
        model = ExponentialSmoothing(series.values, trend="add", seasonal=None, damped_trend=True)
        fit = model.fit(optimized=True)
        forecast_vals = fit.forecast(horizon)
    except Exception:
        # Régression linéaire de secours
        x = np.arange(len(series))
        coef = np.polyfit(x, series.values, 1)
        forecast_vals = np.polyval(coef, np.arange(len(series), len(series) + horizon))

    # Eviter les valeurs négatives
    forecast_vals = np.maximum(forecast_vals, 0)
    return pd.Series(forecast_vals, index=future_years)


# ─────────────────────────────────────────────
# APPLICATION PRINCIPALE
# ─────────────────────────────────────────────
def main():
    # ── Header ─────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px 0;'>
        <h1 style='font-size:2.4rem; margin-bottom:4px;'>🛡️ OASIS Security</h1>
        <p style='color:#8b949e; font-size:1rem;'>
            Analyse de la délinquance enregistrée en France · 2016–2025 · Prévisions 2030
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Chargement ─────────────────────────────
    df_raw = load_data()
    geojson = load_geojson()

    if df_raw is None:
        st.error("❌ Données indisponibles. Vérifiez votre connexion ou l'URL du dataset.")
        st.stop()

    mapping = detect_columns(df_raw)

    # Vérification des colonnes minimales
    required = ["dep", "annee", "classe", "valeur"]
    missing = [r for r in required if r not in mapping]
    if missing:
        st.error(
            f"Colonnes non détectées : {missing}. "
            f"Colonnes disponibles dans le CSV : {list(df_raw.columns[:20])}"
        )
        with st.expander("🔍 Aperçu brut du dataset"):
            st.dataframe(df_raw.head(10))
        st.stop()

    # Noms réels des colonnes
    COL_DEP    = mapping["dep"]
    COL_LIB    = mapping.get("lib_dep", COL_DEP)
    COL_ANNEE  = mapping["annee"]
    COL_CLASSE = mapping["classe"]
    COL_FAITS  = mapping["valeur"]
    COL_TAUX   = mapping.get("taux")
    COL_POP    = mapping.get("pop")

    # Nettoyage léger
    df = df_raw.copy()
    df[COL_FAITS] = pd.to_numeric(df[COL_FAITS].astype(str).str.replace(",", "."), errors="coerce")
    df[COL_ANNEE] = pd.to_numeric(df[COL_ANNEE], errors="coerce")
    df = df.dropna(subset=[COL_FAITS, COL_ANNEE])
    df[COL_ANNEE] = df[COL_ANNEE].astype(int)

    # Listes de valeurs
    all_years   = sorted(df[COL_ANNEE].unique())
    all_crimes  = sorted(df[COL_CLASSE].dropna().unique())
    all_deps    = sorted(df[COL_DEP].dropna().astype(str).unique())

    # Libellés département
    if COL_LIB != COL_DEP:
        dep_labels = (
            df[[COL_DEP, COL_LIB]]
            .drop_duplicates()
            .set_index(COL_DEP.strip())[COL_LIB]
            .to_dict()
        )
    else:
        dep_labels = {d: d for d in all_deps}

    # ── SIDEBAR ────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.shields.io/badge/OASIS-Security-blue?style=for-the-badge&logo=shield&logoColor=white",
            use_container_width=True,
        )
        st.markdown("---")
        st.subheader("🗂️ Filtres")

        # Sélection département
        dep_options = ["🇫🇷 France entière"] + [
            f"{d} – {dep_labels.get(d, d)}" for d in all_deps
        ]
        dep_selection = st.multiselect(
            "Département(s)",
            options=dep_options,
            default=["🇫🇷 France entière"],
        )

        # Sélection crimes
        crime_selection = st.multiselect(
            "Crime(s) / Délit(s)",
            options=all_crimes,
            default=all_crimes[:3] if len(all_crimes) >= 3 else all_crimes,
        )

        st.markdown("---")
        st.markdown("**📊 Options carte**")
        map_year = st.select_slider(
            "Année de la carte",
            options=all_years,
            value=max(all_years),
        )
        if crime_selection:
            map_crime = st.selectbox("Indicateur carte", options=crime_selection)
        else:
            map_crime = all_crimes[0] if all_crimes else None

        st.markdown("---")
        st.caption("Données : [data.gouv.fr](https://www.data.gouv.fr)")
        st.caption("CDSD – Data Science Project")

    # ─── Filtrage ──────────────────────────────
    france_only = "🇫🇷 France entière" in dep_selection or not dep_selection

    if france_only:
        df_filtered = df.copy()
        scope_label = "France entière"
    else:
        selected_dep_codes = [s.split(" – ")[0] for s in dep_selection if s != "🇫🇷 France entière"]
        df_filtered = df[df[COL_DEP].astype(str).isin(selected_dep_codes)]
        scope_label = ", ".join(selected_dep_codes)

    if crime_selection:
        df_filtered = df_filtered[df_filtered[COL_CLASSE].isin(crime_selection)]

    # ─────────────────────────────────────────────
    # KPIs – Ligne du haut
    # ─────────────────────────────────────────────
    st.markdown("### 📈 Indicateurs clés")
    col1, col2, col3, col4 = st.columns(4)

    year_max = max(all_years)
    year_prev = year_max - 1 if year_max - 1 in all_years else year_max

    total_last  = df_filtered[df_filtered[COL_ANNEE] == year_max][COL_FAITS].sum()
    total_prev  = df_filtered[df_filtered[COL_ANNEE] == year_prev][COL_FAITS].sum()
    delta_pct   = ((total_last - total_prev) / total_prev * 100) if total_prev > 0 else 0

    nb_crimes   = df_filtered[COL_CLASSE].nunique()
    nb_deps_sel = df_filtered[COL_DEP].nunique()

    total_all_years = df_filtered[COL_FAITS].sum()

    col1.metric(f"Faits en {year_max}", f"{total_last:,.0f}".replace(",", " "),
                delta=f"{delta_pct:+.1f}% vs {year_prev}")
    col2.metric("Cumul toutes années", f"{total_all_years:,.0f}".replace(",", " "))
    col3.metric("Types de crimes/délits", nb_crimes)
    col4.metric("Départements couverts", nb_deps_sel)

    st.markdown("---")

    # ─────────────────────────────────────────────
    # GRAPHIQUE 1 – Évolution + Prévision
    # ─────────────────────────────────────────────
    st.markdown("### 📉 Évolution 2016–2025 & Prévisions jusqu'en 2030")

    if not crime_selection:
        st.info("Sélectionnez au moins un crime/délit dans la barre latérale.")
    else:
        fig_evol = go.Figure()

        for crime in crime_selection:
            df_crime = df_filtered[df_filtered[COL_CLASSE] == crime]
            serie = (
                df_crime.groupby(COL_ANNEE)[COL_FAITS]
                .sum()
                .sort_index()
            )
            if serie.empty:
                continue

            # Historique
            fig_evol.add_trace(go.Scatter(
                x=serie.index,
                y=serie.values,
                mode="lines+markers",
                name=crime,
                line=dict(width=2),
                marker=dict(size=6),
            ))

            # Prévision
            last_year = int(serie.index[-1])
            horizon   = 2030 - last_year
            if horizon > 0:
                forecast = forecast_series(serie, horizon=horizon)
                if not forecast.empty:
                    # Pont entre historique et prévision
                    bridge_x = [last_year] + list(forecast.index)
                    bridge_y = [serie.iloc[-1]] + list(forecast.values)
                    fig_evol.add_trace(go.Scatter(
                        x=bridge_x,
                        y=bridge_y,
                        mode="lines",
                        name=f"{crime} (prévision)",
                        line=dict(width=2, dash="dot"),
                        showlegend=True,
                    ))

        # Ligne verticale 2025
        fig_evol.add_vline(
            x=year_max,
            line_dash="dash",
            line_color="rgba(255,255,255,0.3)",
            annotation_text="Dernier relevé",
            annotation_position="top right",
        )

        fig_evol.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            legend=dict(orientation="h", y=-0.2),
            xaxis_title="Année",
            yaxis_title="Nombre de faits",
            height=480,
            margin=dict(l=40, r=20, t=30, b=80),
            hovermode="x unified",
        )
        st.plotly_chart(fig_evol, use_container_width=True)

    # ─────────────────────────────────────────────
    # GRAPHIQUE 2 – Carte choroplèthe
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🗺️ Carte interactive par département – {map_year}")

    if geojson is None:
        st.warning("GeoJSON non chargé. La carte est indisponible.")
    elif map_crime is None:
        st.info("Sélectionnez un indicateur pour la carte.")
    else:
        df_map = (
            df[(df[COL_ANNEE] == map_year) & (df[COL_CLASSE] == map_crime)]
            .groupby(COL_DEP)[COL_FAITS]
            .sum()
            .reset_index()
        )
        df_map.columns = ["dep", "faits"]
        df_map["dep"] = df_map["dep"].astype(str).str.zfill(2)

        # Calcul du rang et du pourcentage
        total_france = df_map["faits"].sum()
        df_map["pct"] = (df_map["faits"] / total_france * 100).round(2)
        df_map["rang"] = df_map["faits"].rank(ascending=False).astype(int)

        fig_map = px.choropleth(
            df_map,
            geojson=geojson,
            locations="dep",
            featureidkey="properties.code",
            color="faits",
            color_continuous_scale="Reds",
            hover_data={"faits": True, "pct": True, "rang": True},
            labels={"faits": "Faits", "pct": "% national", "rang": "Rang"},
        )
        fig_map.update_geos(
            fitbounds="locations",
            visible=False,
            bgcolor="#0d1117",
        )
        fig_map.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            coloraxis_colorbar=dict(title="Faits"),
            height=550,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # ─────────────────────────────────────────────
    # GRAPHIQUE 3 – Top départements
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🏆 Top départements par volume de faits – {year_max}")

    if crime_selection:
        df_top = (
            df[(df[COL_ANNEE] == year_max) & (df[COL_CLASSE].isin(crime_selection))]
            .groupby(COL_DEP)[COL_FAITS]
            .sum()
            .reset_index()
            .sort_values(COL_FAITS, ascending=False)
            .head(20)
        )
        df_top.columns = ["Département", "Faits"]
        total_top = df_top["Faits"].sum()
        df_top["% national"] = (df_top["Faits"] / df[(df[COL_ANNEE] == year_max) & (df[COL_CLASSE].isin(crime_selection))][COL_FAITS].sum() * 100).round(2)
        df_top["Libellé"] = df_top["Département"].astype(str).map(lambda x: dep_labels.get(x, x))

        col_a, col_b = st.columns([2, 1])

        with col_a:
            fig_bar = px.bar(
                df_top,
                x="Faits",
                y="Libellé",
                orientation="h",
                color="Faits",
                color_continuous_scale="Blues",
                text="Faits",
                hover_data={"% national": True},
            )
            fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_bar.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0d1117",
                plot_bgcolor="#161b22",
                font=dict(color="#e6edf3"),
                yaxis=dict(autorange="reversed"),
                showlegend=False,
                height=500,
                margin=dict(l=10, r=60, t=20, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_b:
            st.markdown("**Tableau détaillé**")
            df_display = df_top[["Département", "Libellé", "Faits", "% national"]].copy()
            df_display["Faits"] = df_display["Faits"].apply(lambda x: f"{x:,.0f}".replace(",", " "))
            df_display["% national"] = df_display["% national"].apply(lambda x: f"{x:.2f}%")
            df_display.index = range(1, len(df_display) + 1)
            st.dataframe(df_display, use_container_width=True, height=480)

    # ─────────────────────────────────────────────
    # GRAPHIQUE 4 – Répartition par type de crime
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🍕 Répartition par type de crime – {year_max} – {scope_label}")

    df_pie = (
        df_filtered[df_filtered[COL_ANNEE] == year_max]
        .groupby(COL_CLASSE)[COL_FAITS]
        .sum()
        .reset_index()
        .sort_values(COL_FAITS, ascending=False)
    )
    if not df_pie.empty:
        # Regrouper les petites catégories
        top_n = 10
        if len(df_pie) > top_n:
            others = df_pie.iloc[top_n:][COL_FAITS].sum()
            df_pie = df_pie.head(top_n)
            df_pie = pd.concat([df_pie, pd.DataFrame({COL_CLASSE: ["Autres"], COL_FAITS: [others]})])

        fig_pie = px.pie(
            df_pie,
            names=COL_CLASSE,
            values=COL_FAITS,
            color_discrete_sequence=px.colors.sequential.Blues_r,
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            legend=dict(orientation="v", x=1.02),
            height=460,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ─────────────────────────────────────────────
    # SECTION DONNÉES BRUTES
    # ─────────────────────────────────────────────
    with st.expander("🗃️ Données brutes filtrées"):
        st.dataframe(df_filtered.sort_values([COL_ANNEE, COL_DEP]).head(500), use_container_width=True)
        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Télécharger CSV filtré",
            data=csv,
            file_name="oasis_security_filtered.csv",
            mime="text/csv",
        )

    # ─────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center; color:#8b949e; font-size:0.85rem; padding: 10px 0;'>
            🛡️ <strong>OASIS Security</strong> · Données : 
            <a href='https://www.data.gouv.fr' style='color:#58a6ff;'>data.gouv.fr</a> · 
            Police & Gendarmerie Nationales · 2016–2025 · 
            Prévisions via Holt-Winters (statsmodels)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
