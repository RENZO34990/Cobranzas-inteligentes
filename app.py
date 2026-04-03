"""
========================================================
  SISTEMA DE COBRANZA INTELIGENTE CON IA
  Dashboard interactivo con Streamlit
  Portafolio CV | Data Analyst / Risk Analyst
========================================================

INSTRUCCIONES DE USO:
1. Instala dependencias:
   pip install streamlit pandas numpy scikit-learn plotly

2. Entrena el modelo primero en Google Colab y guarda:
   import joblib
   joblib.dump(modelo, 'modelo_cobranza.pkl')
   joblib.dump(X_train.columns.tolist(), 'features.pkl')

3. Pon modelo_cobranza.pkl y features.pkl en la misma carpeta que este archivo

4. Ejecuta:
   streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Intento de carga del modelo real ──────────────────────────────────────────
MODEL_LOADED = False
try:
    import joblib, os
    if os.path.exists('modelo_cobranza.pkl') and os.path.exists('features.pkl'):
        modelo   = joblib.load('modelo_cobranza.pkl')
        features = joblib.load('features.pkl')
        MODEL_LOADED = True
except Exception:
    pass

# ── Configuración de la página ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard de Cobranza Inteligente",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Estilos CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        border: 1px solid #e9ecef;
    }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-label { color: #6c757d; font-size: 0.85rem; margin: 0; }
    .badge-alto   { background:#d4edda; color:#155724; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-medio  { background:#fff3cd; color:#856404; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .badge-bajo   { background:#f8d7da; color:#721c24; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
    .stDataFrame  { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Funciones de segmentación ──────────────────────────────────────────────────
def segmentar(p):
    if p >= 0.75: return "Alta"
    if p >= 0.45: return "Media"
    return "Baja"

def accion(p):
    if p >= 0.75: return "Llamada inmediata"
    if p >= 0.45: return "Email / SMS"
    return "Derivar a cobranza especializada"

def prioridad(p):
    if p >= 0.75: return 1
    if p >= 0.45: return 2
    return 3

# ── Generación de datos demo ────────────────────────────────────────────────────
@st.cache_data
def generar_datos_demo(n=200, seed=42):
    np.random.seed(seed)
    sev   = np.random.randint(0, 11, n)
    deuda = np.round(np.random.uniform(0.05, 1.8, n), 2)
    edad  = np.random.randint(22, 69, n)
    ing   = np.random.randint(1500, 15000, n)
    dep   = np.random.randint(0, 4, n)
    util  = np.round(np.random.uniform(0, 1, n), 2)

    prob_base = np.clip(0.85 - sev * 0.06 - deuda * 0.12 + np.random.uniform(-0.1, 0.1, n), 0.02, 0.98)
    prob      = np.round(prob_base, 3)

    df = pd.DataFrame({
        'cliente_id':           [f'CLI-{str(i+1).zfill(4)}' for i in range(n)],
        'prob_pago':            prob,
        'prob_mora':            np.round(1 - prob, 3),
        'segmento':             [segmentar(p) for p in prob],
        'accion_recomendada':   [accion(p) for p in prob],
        'prioridad':            [prioridad(p) for p in prob],
        'ratio_deuda_ingreso':  deuda,
        'severidad_mora':       sev,
        'edad':                 edad,
        'ingreso_mensual':      ing,
        'dependientes':         dep,
        'util_lineas_rotativas': util,
    })
    return df.sort_values('prioridad').reset_index(drop=True)


@st.cache_data
def predecir_con_modelo(df_input):
    """Usa el modelo real si está disponible, si no devuelve None."""
    if not MODEL_LOADED:
        return None
    try:
        X = df_input[features]
        proba = modelo.predict_proba(X)[:, 1]
        return np.round(1 - proba, 3)   # prob_pago = 1 - prob_mora
    except Exception:
        return None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏦 Cobranza IA")
    st.markdown("---")

    st.subheader("Fuente de datos")
    fuente = st.radio(
        "Selecciona",
        ["Datos de demo", "Subir CSV propio"],
        help="Usa datos de demo para explorar el dashboard. Sube tu propio CSV para usar con el modelo real."
    )

    st.markdown("---")
    st.subheader("Filtros")

    seg_filter = st.multiselect(
        "Segmento",
        ["Alta", "Media", "Baja"],
        default=["Alta", "Media", "Baja"]
    )

    accion_filter = st.multiselect(
        "Acción recomendada",
        ["Llamada inmediata", "Email / SMS", "Derivar a cobranza especializada"],
        default=["Llamada inmediata", "Email / SMS", "Derivar a cobranza especializada"]
    )

    prob_range = st.slider(
        "Rango de probabilidad de pago",
        min_value=0.0, max_value=1.0,
        value=(0.0, 1.0), step=0.05,
        format="%.0f%%"
    )

    st.markdown("---")
    n_clientes = st.slider("Clientes en demo", 50, 500, 200, 50)

    st.markdown("---")
    if MODEL_LOADED:
        st.success("Modelo cargado correctamente")
    else:
        st.info("Usando datos simulados.\n\nEntrena el modelo en Colab y coloca `modelo_cobranza.pkl` en esta carpeta para activar predicciones reales.")

# ── Carga de datos ─────────────────────────────────────────────────────────────
if fuente == "Subir CSV propio":
    uploaded = st.file_uploader("Sube tu CSV de clientes", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        prob_predicha = predecir_con_modelo(df_raw)
        if prob_predicha is not None:
            df_raw['prob_pago']          = prob_predicha
            df_raw['prob_mora']          = np.round(1 - prob_predicha, 3)
            df_raw['segmento']           = df_raw['prob_pago'].apply(segmentar)
            df_raw['accion_recomendada'] = df_raw['prob_pago'].apply(accion)
            df_raw['prioridad']          = df_raw['prob_pago'].apply(prioridad)
            if 'cliente_id' not in df_raw.columns:
                df_raw['cliente_id'] = [f'CLI-{str(i+1).zfill(4)}' for i in range(len(df_raw))]
            df = df_raw.sort_values('prioridad').reset_index(drop=True)
            st.success(f"Predicciones generadas para {len(df)} clientes con modelo real.")
        else:
            st.warning("No se pudo usar el modelo. Asegúrate de que el CSV tiene las columnas correctas. Mostrando datos demo.")
            df = generar_datos_demo(n_clientes)
    else:
        df = generar_datos_demo(n_clientes)
else:
    df = generar_datos_demo(n_clientes)

# ── Aplicar filtros ────────────────────────────────────────────────────────────
df_f = df[
    df['segmento'].isin(seg_filter) &
    df['accion_recomendada'].isin(accion_filter) &
    df['prob_pago'].between(prob_range[0], prob_range[1])
].copy()

# ── Título principal ───────────────────────────────────────────────────────────
st.title("Dashboard de Cobranza Inteligente")
st.caption(f"Mostrando {len(df_f):,} clientes · Datos {'reales (modelo entrenado)' if MODEL_LOADED else 'simulados (demo)'}")
st.markdown("---")

# ── KPIs ───────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

total    = len(df_f)
avg_prob = df_f['prob_pago'].mean() * 100 if total > 0 else 0
n_alto   = len(df_f[df_f['segmento'] == 'Alta'])
n_bajo   = len(df_f[df_f['segmento'] == 'Baja'])

with col1:
    st.metric("Total clientes", f"{total:,}", help="Clientes en cartera activa filtrada")
with col2:
    st.metric("Prob. pago promedio", f"{avg_prob:.1f}%", help="Promedio ponderado de la cartera")
with col3:
    st.metric("Alta probabilidad", f"{n_alto:,}", delta=f"{n_alto/total*100:.0f}% del total" if total else None,
              delta_color="normal", help="Candidatos para llamada inmediata")
with col4:
    st.metric("Requieren escalamiento", f"{n_bajo:,}",
              delta=f"-{n_bajo/total*100:.0f}% del total" if total else None,
              delta_color="inverse", help="Derivar a cobranza especializada")

st.markdown("---")

# ── Fila principal: tabla + donut ──────────────────────────────────────────────
col_tabla, col_donut = st.columns([2, 1])

with col_tabla:
    st.subheader("Ranking de clientes — prioridad de contacto")

    sort_opt = st.selectbox(
        "Ordenar por",
        ["Probabilidad de pago (mayor a menor)",
         "Probabilidad de mora (mayor a menor)",
         "Ratio deuda/ingreso",
         "Severidad de mora"],
        label_visibility="collapsed"
    )

    df_show = df_f.copy()
    if "pago" in sort_opt:
        df_show = df_show.sort_values('prob_pago', ascending=False)
    elif "mora" in sort_opt and "ratio" not in sort_opt and "Sev" not in sort_opt:
        df_show = df_show.sort_values('prob_mora', ascending=False)
    elif "ratio" in sort_opt.lower():
        df_show = df_show.sort_values('ratio_deuda_ingreso', ascending=False)
    else:
        df_show = df_show.sort_values('severidad_mora', ascending=False)

    df_display = df_show[[
        'cliente_id', 'prob_pago', 'segmento',
        'accion_recomendada', 'ratio_deuda_ingreso', 'severidad_mora'
    ]].head(20).copy()

    df_display.columns = [
        'Cliente ID', 'Prob. Pago', 'Segmento',
        'Acción', 'Deuda/Ing', 'Sev. Mora'
    ]
    df_display['Prob. Pago'] = (df_display['Prob. Pago'] * 100).round(1).astype(str) + '%'

    def color_segmento(val):
        c = {'Alta': 'background-color: #d4edda; color: #155724',
             'Media': 'background-color: #fff3cd; color: #856404',
             'Baja': 'background-color: #f8d7da; color: #721c24'}
        return c.get(val, '')

    st.dataframe(
        df_display.style.applymap(color_segmento, subset=['Segmento']),
        use_container_width=True,
        height=380,
        hide_index=True
    )

with col_donut:
    st.subheader("Distribución por segmento")

    seg_counts = df_f['segmento'].value_counts().reindex(['Alta', 'Media', 'Baja'], fill_value=0)

    fig_donut = go.Figure(go.Pie(
        labels=seg_counts.index,
        values=seg_counts.values,
        hole=0.55,
        marker_colors=['#3B6D11', '#EF9F27', '#E24B4A'],
        textinfo='percent+label',
        textfont_size=12,
        hovertemplate='<b>%{label}</b><br>Clientes: %{value}<br>%{percent}<extra></extra>'
    ))
    fig_donut.add_annotation(
        text=f"<b>{total}</b><br>clientes",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14)
    )
    fig_donut.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        showlegend=False
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    st.markdown("**Acciones recomendadas**")
    for seg, color, accion_txt in [
        ("Alta", "#3B6D11", "Llamada inmediata"),
        ("Media", "#BA7517", "Email / SMS"),
        ("Baja", "#A32D2D", "Derivar a especialista"),
    ]:
        n = seg_counts.get(seg, 0)
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:6px 0;border-bottom:1px solid #eee">'
            f'<span style="color:{color};font-weight:600;font-size:13px">{accion_txt}</span>'
            f'<span style="font-weight:700;font-size:14px">{n}</span></div>',
            unsafe_allow_html=True
        )

st.markdown("---")

# ── Fila gráficas: histograma + scatter ────────────────────────────────────────
col_hist, col_scatter = st.columns(2)

with col_hist:
    st.subheader("Distribución de probabilidad de pago")

    fig_hist = px.histogram(
        df_f,
        x='prob_pago',
        nbins=20,
        color='segmento',
        color_discrete_map={'Alta': '#3B6D11', 'Media': '#EF9F27', 'Baja': '#E24B4A'},
        labels={'prob_pago': 'Probabilidad de pago', 'count': 'N° clientes'},
        category_orders={'segmento': ['Alta', 'Media', 'Baja']}
    )
    fig_hist.update_layout(
        margin=dict(t=10, b=30, l=0, r=0),
        height=280,
        legend=dict(orientation='h', y=-0.25),
        bargap=0.05
    )
    fig_hist.update_xaxes(tickformat='.0%')
    st.plotly_chart(fig_hist, use_container_width=True)

with col_scatter:
    st.subheader("Severidad de mora vs probabilidad de pago")

    fig_sc = px.scatter(
        df_f.sample(min(len(df_f), 100), random_state=1),
        x='severidad_mora',
        y='prob_pago',
        color='segmento',
        color_discrete_map={'Alta': '#3B6D11', 'Media': '#EF9F27', 'Baja': '#E24B4A'},
        size='ratio_deuda_ingreso',
        hover_data=['cliente_id', 'accion_recomendada'],
        labels={
            'severidad_mora': 'Severidad de mora',
            'prob_pago': 'Probabilidad de pago',
        },
        category_orders={'segmento': ['Alta', 'Media', 'Baja']}
    )
    fig_sc.update_layout(
        margin=dict(t=10, b=30, l=0, r=0),
        height=280,
        legend=dict(orientation='h', y=-0.25)
    )
    fig_sc.update_yaxes(tickformat='.0%')
    st.plotly_chart(fig_sc, use_container_width=True)

st.markdown("---")

# ── Análisis de variables ──────────────────────────────────────────────────────
st.subheader("Análisis de variables por segmento")

cols_analisis = ['prob_pago', 'ratio_deuda_ingreso', 'severidad_mora', 'edad', 'ingreso_mensual']
resumen = df_f.groupby('segmento')[cols_analisis].mean().round(3)
resumen.columns = ['Prob. Pago', 'Ratio Deuda/Ing', 'Sev. Mora (avg)', 'Edad Prom.', 'Ingreso Prom.']
resumen['Prob. Pago'] = (resumen['Prob. Pago'] * 100).round(1).astype(str) + '%'
resumen['Ingreso Prom.'] = resumen['Ingreso Prom.'].apply(lambda x: f"S/. {x:,.0f}")

st.dataframe(resumen.style.highlight_max(axis=0, color='#d4edda', subset=['Ratio Deuda/Ing', 'Sev. Mora (avg)'])
             .highlight_min(axis=0, color='#f8d7da', subset=['Ratio Deuda/Ing', 'Sev. Mora (avg)']),
             use_container_width=True)

st.markdown("---")

# ── Exportar ───────────────────────────────────────────────────────────────────
st.subheader("Exportar cartera priorizada")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    csv = df_f.sort_values('prioridad').to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV — cartera completa",
        data=csv,
        file_name='cartera_priorizada.csv',
        mime='text/csv',
        use_container_width=True
    )

with col_dl2:
    csv_top = df_f[df_f['segmento'] == 'Alta'].sort_values('prob_pago', ascending=False).to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar CSV — solo alta probabilidad",
        data=csv_top,
        file_name='clientes_alta_prioridad.csv',
        mime='text/csv',
        use_container_width=True
    )

st.caption("Sistema de Cobranza Inteligente · Desarrollado con Python, scikit-learn y Streamlit")
