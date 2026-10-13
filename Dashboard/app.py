from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px

# ==================================================
# CARGA DE DATOS
# ==================================================

df = pd.read_csv(
    "../DatasetOilCostProjectWorldwide.csv",
    sep=None,
    engine="python"
)

# ==================================================
# PREPARACIÓN DE DATOS
# ==================================================

target = "BrentEquivalentProductionCost"

df_target = df.dropna(subset=[target])

# ==================================================
# KPIs
# ==================================================

n_obs = len(df)
n_assets = df["CostProductionElementName"].nunique()
n_countries = df["CountryName"].nunique()
n_regions = df["RegionName"].nunique()

# ==================================================
# FIGURA 1 - REGIONES
# ==================================================

region_counts = (
    df["RegionName"]
    .value_counts()
    .reset_index()
)

region_counts.columns = ["Region", "Count"]

fig_region = px.bar(
    region_counts,
    x="Region",
    y="Count",
    title="Distribución de observaciones por región"
)

# ==================================================
# FIGURA 2 - EVOLUCIÓN TEMPORAL
# ==================================================

yearly_cost = (
    df_target
    .groupby("Year")[target]
    .mean()
    .reset_index()
)

fig_time = px.line(
    yearly_cost,
    x="Year",
    y=target,
    markers=True,
    title="Evolución temporal promedio del BrentEquivalentProductionCost"
)

# ==================================================
# FIGURA 3 - HISTOGRAMA
# ==================================================

fig_hist = px.histogram(
    df_target,
    x=target,
    nbins=30,
    title="Distribución del BrentEquivalentProductionCost"
)

# ==================================================
# FIGURA 4 - BOXPLOT
# ==================================================

fig_box = px.box(
    df_target,
    y=target,
    title="Boxplot del BrentEquivalentProductionCost"
)

# ==================================================
# FIGURA 5 - HEATMAP
# ==================================================

num_cols = [
    "BrentDifferential",
    "BrentEquivalentProductionCost",
    "WellheadDevelopmentCost",
    "BrentEquivalentDevelopmentCost",
    "WellheadProductionCost",
    "Production"
]

corr = (
    df_target[num_cols]
    .corr(method="spearman")
)

fig_corr = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    title="Matriz de Correlación (Spearman)"
)

# ==================================================
# FIGURA 6 - PCA
# ==================================================

variance_df = pd.DataFrame(
    {
        "Componente": [
            "PC1",
            "PC2",
            "PC3",
            "PC4",
            "PC5"
        ],
        "Varianza": [
            0.5467,
            0.2150,
            0.1627,
            0.0756,
            0.0000
        ]
    }
)

fig_pca = px.bar(
    variance_df,
    x="Componente",
    y="Varianza",
    title="Varianza Explicada por Componentes Principales"
)

# ==================================================
# FIGURA 7 - MODELO
# ==================================================

results = pd.DataFrame(
    {
        "Modelo": [
            "Dummy Regressor",
            "SVR Lineal"
        ],
        "RMSE": [
            5.82,
            2.19
        ],
        "MAPE": [
            31.87,
            9.45
        ],
        "R2": [
            -0.004,
            0.857
        ]
    }
)

fig_model = px.bar(
    results,
    x="Modelo",
    y="R2",
    color="Modelo",
    title="Comparación de desempeño (R²)"
)

# ==================================================
# ESTILOS
# ==================================================

card_style = {
    "backgroundColor": "white",
    "padding": "15px",
    "borderRadius": "10px",
    "boxShadow": "0px 3px 8px rgba(0,0,0,0.15)",
    "textAlign": "center",
    "width": "18%",
    "display": "inline-block",
    "marginRight": "10px"
}

model_card = {
    "backgroundColor": "white",
    "padding": "15px",
    "borderRadius": "10px",
    "boxShadow": "0px 3px 8px rgba(0,0,0,0.15)",
    "width": "25%",
    "display": "inline-block",
    "textAlign": "center",
    "marginRight": "20px"
}

# ==================================================
# APP
# ==================================================

app = Dash(__name__)

app.layout = html.Div(

    [

        html.H1(
            "Predicción del BrentEquivalentProductionCost en activos Upstream de Oil & Gas",
            style={
                "textAlign": "center"
            }
        ),

        dcc.Tabs(

            [

                # ==================================================
                # TAB 1
                # ==================================================

                dcc.Tab(
                    label="1. Contexto del Problema",

                    children=[

                        html.Br(),

                        html.Img(
                            src="/assets/hero_oilfield.jpg",
                            style={
                                "width": "100%",
                                "height": "350px",
                                "objectFit": "cover",
                                "borderRadius": "10px"
                            }
                        ),

                        html.Br(),

                        html.H2(
                            "Predicción del BrentEquivalentProductionCost"
                        ),

                        html.P(
                            """
                            Este proyecto desarrolla un modelo de Machine Learning
                            para estimar el BrentEquivalentProductionCost de activos
                            upstream de petróleo utilizando variables geográficas,
                            operativas y temporales.
                            """
                        ),

                        html.H3("Descripción del Dataset"),

                        html.Div(

                            [

                                html.Div(
                                    [
                                        html.H3(f"{n_obs:,}"),
                                        html.P("Observaciones")
                                    ],
                                    style=card_style
                                ),

                                html.Div(
                                    [
                                        html.H3(f"{n_assets}"),
                                        html.P("Activos")
                                    ],
                                    style=card_style
                                ),

                                html.Div(
                                    [
                                        html.H3(f"{n_countries}"),
                                        html.P("Países")
                                    ],
                                    style=card_style
                                ),

                                html.Div(
                                    [
                                        html.H3(f"{n_regions}"),
                                        html.P("Regiones")
                                    ],
                                    style=card_style
                                ),

                                html.Div(
                                    [
                                        html.H3("2014-2023"),
                                        html.P("Periodo")
                                    ],
                                    style=card_style
                                ),

                            ]

                        ),

                        html.Br(),

                        html.H3("Pregunta de Investigación"),

                        html.P(
                            """
                            ¿Es posible predecir el BrentEquivalentProductionCost
                            utilizando variables operativas, económicas y geográficas
                            mediante técnicas de Machine Learning?
                            """
                        ),

                        dcc.Graph(figure=fig_region),

                        dcc.Graph(figure=fig_time)

                    ]
                ),

                # ==================================================
                # TAB 2
                # ==================================================

                dcc.Tab(
                    label="2. EDA",

                    children=[

                        html.Br(),

                        html.H2(
                            "Análisis Exploratorio de Datos"
                        ),

                        dcc.Graph(
                            figure=fig_hist
                        ),

                        dcc.Graph(
                            figure=fig_box
                        ),

                        dcc.Graph(
                            figure=fig_corr
                        ),

                        dcc.Graph(
                            figure=fig_pca
                        )

                    ]
                ),

                # ==================================================
                # TAB 3
                # ==================================================

                dcc.Tab(
                    label="3. Modelos Base",

                    children=[

                        html.Br(),

                        html.H2(
                            "Resultados del Modelo Base"
                        ),

                        html.Div(

                            [

                                html.Div(
                                    [
                                        html.H2("0.857"),
                                        html.P("R²")
                                    ],
                                    style=model_card
                                ),

                                html.Div(
                                    [
                                        html.H2("2.19"),
                                        html.P("RMSE")
                                    ],
                                    style=model_card
                                ),

                                html.Div(
                                    [
                                        html.H2("9.45%"),
                                        html.P("MAPE")
                                    ],
                                    style=model_card
                                )

                            ]

                        ),

                        html.Br(),

                        dcc.Graph(
                            figure=fig_model
                        ),

                        html.H3("Comparación de Modelos"),

                        html.Table(

                            [

                                html.Tr(
                                    [
                                        html.Th("Modelo"),
                                        html.Th("RMSE"),
                                        html.Th("MAPE"),
                                        html.Th("R²")
                                    ]
                                ),

                                html.Tr(
                                    [
                                        html.Td("Dummy Regressor"),
                                        html.Td("5.82"),
                                        html.Td("31.87%"),
                                        html.Td("-0.004")
                                    ]
                                ),

                                html.Tr(
                                    [
                                        html.Td("SVR Lineal"),
                                        html.Td("2.19"),
                                        html.Td("9.45%"),
                                        html.Td("0.857")
                                    ]
                                )

                            ],

                            style={
                                "width": "60%",
                                "margin": "auto",
                                "textAlign": "center"
                            }

                        ),

                        html.Br(),

                        html.H3(
                            "Bootstrap (95%)"
                        ),

                        html.Ul(
                            [
                                html.Li(
                                    "RMSE: [1.96 , 2.48]"
                                ),
                                html.Li(
                                    "MAPE: [8.84% , 10.07%]"
                                ),
                                html.Li(
                                    "R²: [0.837 , 0.878]"
                                )
                            ]
                        )

                    ]
                )

            ]

        ),
        html.Br(),

html.H3("Conclusión del Modelo"),

html.Div(

    [

        html.P(
            """
            El modelo SVR lineal superó ampliamente el desempeño de la
            línea base Dummy Regressor en todas las métricas evaluadas.
            """
        ),

        html.P(
            """
            El error porcentual promedio (MAPE) se redujo desde
            aproximadamente 31.9% hasta 9.45%, mientras que el
            coeficiente de determinación alcanzó un valor de R² = 0.857.
            """
        ),

        html.P(
            """
            Los intervalos de confianza obtenidos mediante bootstrap
            muestran resultados estables y consistentes para la muestra
            analizada.
            """
        ),

        html.P(
            """
            El análisis de residuos evidenció una distribución centrada
            alrededor de cero, ausencia de sesgos sistemáticos y una
            autocorrelación temporal limitada.
            """
        ),

        html.P(
            """
            En conjunto, los resultados indican que las variables
            geográficas, operativas y temporales incluidas en el estudio
            contienen información relevante para explicar el
            BrentEquivalentProductionCost de activos upstream de
            petróleo.
            """
        )

    ],

    style={
        "backgroundColor": "white",
        "padding": "20px",
        "borderRadius": "10px",
        "boxShadow": "0px 3px 8px rgba(0,0,0,0.15)",
        "marginTop": "20px"
    }

)
    ],

    style={
        "padding": "20px",
        "backgroundColor": "#f4f6f9"
    }

)

if __name__ == "__main__":
    app.run(debug=False)