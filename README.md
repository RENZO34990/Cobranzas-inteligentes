# Sistema de Cobranza Inteligente con IA

Dashboard interactivo que predice la probabilidad de pago de clientes morosos y prioriza las acciones de cobranza usando machine learning.

---

## Descripción del proyecto

Las entidades financieras y fintechs enfrentan alta morosidad con estrategias de cobranza genéricas que desperdician tiempo y recursos. Este proyecto construye un sistema que responde a la pregunta clave:

> **¿A quién cobrar primero para maximizar la recuperación?**

El sistema clasifica a cada cliente según su probabilidad de pago y recomienda la acción óptima: llamada inmediata, email/SMS o derivación a cobranza especializada.

---

## Demo

![Dashboard](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)

```bash
streamlit run app.py
```

---

## Estructura del proyecto

```
cobranza-inteligente/
│
├── app.py                  # Dashboard interactivo (Streamlit)
├── notebook.ipynb          # Pipeline completo: EDA → modelo → evaluación
├── modelo_cobranza.pkl     # Modelo entrenado (generado por el notebook)
├── features.pkl            # Lista de features del modelo
├── requirements.txt        # Dependencias del proyecto
└── README.md
```

---

## Pipeline del proyecto

### 1. Datos
- Dataset: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle)
- 150,000 clientes con historial crediticio real
- Variable objetivo: mora grave en los próximos 2 años

### 2. EDA y limpieza
- Análisis de distribuciones y correlaciones
- Imputación de valores nulos con mediana
- Eliminación de outliers extremos en ingreso

### 3. Feature engineering
| Variable creada | Fórmula | Lógica |
|---|---|---|
| `ratio_deuda_ingreso` | deuda / ingreso | Carga financiera relativa |
| `frecuencia_mora` | suma de veces en mora | Historial de incumplimiento |
| `severidad_mora` | mora_30×1 + mora_60×2 + mora_90×3 | Gravedad ponderada |
| `ingreso_por_dependiente` | ingreso / (dependientes + 1) | Capacidad real de pago |

### 4. Modelo
- Algoritmo: Random Forest Classifier
- Manejo de desbalance: `class_weight='balanced'`
- Métrica principal: ROC-AUC

### 5. Segmentación estratégica

| Segmento | Prob. pago | Acción recomendada |
|---|---|---|
| Alta probabilidad | ≥ 75% | Llamada inmediata |
| Probabilidad media | 45% – 74% | Email / SMS |
| Baja probabilidad | < 45% | Cobranza especializada |

---

## Resultados

| Métrica | Valor |
|---|---|
| ROC-AUC | ~0.85 |
| Precisión clase mora | ~0.72 |
| Recall clase mora | ~0.68 |

---

## Instalación y uso

### Requisitos
```bash
pip install -r requirements.txt
```

### Opción 1 — Solo el dashboard (datos demo)
```bash
streamlit run app.py
```

### Opción 2 — Con modelo real
1. Abre `notebook.ipynb` en Google Colab
2. Descarga el dataset de Kaggle y súbelo a Colab
3. Ejecuta todas las celdas
4. Descarga `modelo_cobranza.pkl` y `features.pkl`
5. Colócalos en la misma carpeta que `app.py`
6. Ejecuta `streamlit run app.py`

El dashboard detecta automáticamente si el modelo existe y activa las predicciones reales.

---

## Funcionalidades del dashboard

- Filtros por segmento, acción recomendada y rango de probabilidad
- Ranking de clientes priorizado por probabilidad de pago
- KPIs en tiempo real: total, promedio, distribución por segmento
- Gráficas interactivas: histograma, scatter y donut chart
- Análisis comparativo por segmento
- Exportación a CSV de la cartera priorizada

---

## Tecnologías

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)

---

## Aplicación en la industria

Este tipo de sistema es aplicable en:

- Bancos y cajas de ahorro
- Fintechs de crédito al consumo
- Empresas de cobranza
- Retail con venta a crédito

---

## Autor

Desarrollado como proyecto de portafolio para roles de Data Analyst, Risk Analyst y Business Analyst en el sector financiero peruano.
