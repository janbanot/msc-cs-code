# Wizualizacja Danych, Lab 11 - Jan Banot
# Dashboard Dash - Monitoring zużycia energii w budynku

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime
import os


# Generowanie przykładowych danych
def generate_energy_data():
    """Generuje dane dotyczące zużycia energii przez cały rok 2025"""
    dates = pd.date_range(start="2025-01-01", end="2025-12-31", freq="H")
    np.random.seed(0)
    data = {
        "timestamp": dates,
        "lighting": np.random.normal(loc=50, scale=5, size=len(dates)),
        "hvac": np.random.normal(loc=100, scale=10, size=len(dates)),
        "devices": np.random.normal(loc=30, scale=3, size=len(dates)),
    }
    return pd.DataFrame(data)


# Inicjalizacja aplikacji Dash
app = Dash(__name__)

# Generowanie lub wczytywanie danych
data_path = os.path.join("sem3", "WD", "data", "energy_data.csv")
if not os.path.exists(data_path):
    print("Generowanie danych...")
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    energy_data = generate_energy_data()
    energy_data.to_csv(data_path, index=False)
    print(f"Dane zapisane do {data_path}")
else:
    print(f"Wczytywanie danych z {data_path}")
    energy_data = pd.read_csv(data_path)
    energy_data["timestamp"] = pd.to_datetime(energy_data["timestamp"])

# Przygotowanie danych
energy_data["total_energy"] = (
    energy_data["lighting"] + energy_data["hvac"] + energy_data["devices"]
)
energy_data["date"] = energy_data["timestamp"].dt.date

# Agregacja danych dziennych dla tabeli
daily_data = (
    energy_data.groupby("date")
    .agg({"lighting": "sum", "hvac": "sum", "devices": "sum", "total_energy": "sum"})
    .reset_index()
)
daily_data.columns = [
    "Data",
    "Oświetlenie (kWh)",
    "HVAC (kWh)",
    "Urządzenia (kWh)",
    "Suma (kWh)",
]
daily_data["Data"] = pd.to_datetime(daily_data["Data"]).dt.strftime("%Y-%m-%d")

# Zakres dat dla suwaka (w formacie timestamp)
min_timestamp = energy_data["timestamp"].min().timestamp()
max_timestamp = energy_data["timestamp"].max().timestamp()


# Funkcja pomocnicza do konwersji timestamp
def timestamp_to_date_str(ts):
    """Konwertuje timestamp na czytelny format daty"""
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")


# Generowanie marks dla suwaka z datami
def generate_slider_marks():
    """Generuje marks dla suwaka z odpowiednimi datami"""
    marks = {}
    start = datetime.datetime.fromtimestamp(min_timestamp)
    end = datetime.datetime.fromtimestamp(max_timestamp)

    # Dodaj znaczniki co miesiąc
    current = start
    while current <= end:
        ts = current.timestamp()
        marks[ts] = current.strftime("%Y-%m-%d")
        # Przejdź do następnego miesiąca
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1, day=1)
        else:
            current = current.replace(month=current.month + 1, day=1)

    return marks


slider_marks = generate_slider_marks()


# Layout aplikacji z CSS Grid
app.layout = html.Div(
    [
        html.H1(
            "Jan Banot - Dashboard - Monitoring Zużycia Energii w Budynku",
            style={
                "textAlign": "center",
                "color": "#2c3e50",
                "marginBottom": "30px",
                "fontFamily": "Arial, sans-serif",
            },
        ),
        # Suwak czasu
        html.Div(
            [
                html.Label(
                    "Filtruj zakres czasu:",
                    style={
                        "fontWeight": "bold",
                        "marginBottom": "10px",
                        "paddingBottom": "55px",
                        "fontSize": "16px",
                    },
                ),
                dcc.RangeSlider(
                    id="time-slider",
                    min=min_timestamp,
                    max=max_timestamp,
                    value=[min_timestamp, max_timestamp],
                    marks=slider_marks,
                    tooltip=None,
                    updatemode="drag",
                ),
                html.Div(
                    id="hover-date-display",
                    style={
                        "textAlign": "center",
                        "marginTop": "5px",
                        "fontSize": "16px",
                        "fontWeight": "bold",
                        "color": "#2c3e50",
                        "minHeight": "25px"
                    },
                ),
            ],
            style={
                "margin": "20px 50px",
                "padding": "20px",
                "backgroundColor": "#ecf0f1",
                "borderRadius": "10px",
            },
        ),
        # Grid layout dla wykresów i tabeli
        html.Div(
            [
                # Pierwszy rząd - Wykres liniowy (pełna szerokość)
                html.Div(
                    [dcc.Graph(id="line-chart")],
                    style={"gridColumn": "1 / -1", "marginBottom": "20px"},
                ),
                # Drugi rząd - Wykres kołowy i histogram
                html.Div([dcc.Graph(id="pie-chart")], style={"gridColumn": "1 / 2"}),
                html.Div([dcc.Graph(id="histogram")], style={"gridColumn": "2 / 3"}),
                # Trzeci rząd - Tabela (pełna szerokość)
                html.Div(
                    [
                        html.H3(
                            "Dzienne zużycie energii",
                            style={"textAlign": "center", "color": "#2c3e50"},
                        ),
                        dash_table.DataTable(
                            id="energy-table",
                            columns=[
                                {"name": col, "id": col} for col in daily_data.columns
                            ],
                            data=daily_data.to_dict("records"),
                            page_size=15,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "center",
                                "padding": "10px",
                                "fontFamily": "Arial, sans-serif",
                            },
                            style_header={
                                "backgroundColor": "#3498db",
                                "color": "white",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#f8f9fa",
                                }
                            ],
                        ),
                    ],
                    style={"gridColumn": "1 / -1", "marginTop": "20px"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "20px",
                "padding": "20px",
                "maxWidth": "1400px",
                "margin": "0 auto",
            },
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f5f6fa",
        "minHeight": "100vh",
        "padding": "20px",
    },
)


# Callback dla interaktywności
@app.callback(
    [
        Output("line-chart", "figure"),
        Output("pie-chart", "figure"),
        Output("histogram", "figure"),
        Output("energy-table", "data"),
        Output("hover-date-display", "children"),
    ],
    [Input("time-slider", "value")],
)
def update_dashboard(time_range):
    """Aktualizuje wszystkie wykresy i tabelę na podstawie wybranego zakresu czasowego"""
    # Konwersja timestamp na datetime
    start_date = datetime.datetime.fromtimestamp(time_range[0])
    end_date = datetime.datetime.fromtimestamp(time_range[1])

    # Filtrowanie danych
    filtered_data = energy_data[
        (energy_data["timestamp"] >= start_date)
        & (energy_data["timestamp"] <= end_date)
    ]

    # 1. Wykres liniowy - zmienność zużycia energii w czasie
    line_fig = go.Figure()
    line_fig.add_trace(
        go.Scatter(
            x=filtered_data["timestamp"],
            y=filtered_data["lighting"],
            mode="lines",
            name="Oświetlenie",
            line=dict(color="#f39c12", width=2),
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=filtered_data["timestamp"],
            y=filtered_data["hvac"],
            mode="lines",
            name="HVAC",
            line=dict(color="#3498db", width=2),
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=filtered_data["timestamp"],
            y=filtered_data["devices"],
            mode="lines",
            name="Urządzenia",
            line=dict(color="#2ecc71", width=2),
        )
    )
    line_fig.add_trace(
        go.Scatter(
            x=filtered_data["timestamp"],
            y=filtered_data["total_energy"],
            mode="lines",
            name="Suma",
            line=dict(color="#e74c3c", width=3, dash="dash"),
        )
    )
    line_fig.update_layout(
        title="Zużycie energii w czasie",
        xaxis_title="Czas",
        yaxis_title="Zużycie energii (kWh)",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    line_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#ecf0f1")
    line_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#ecf0f1")

    # 2. Wykres kołowy - procentowe zużycie energii (całoroczne sumy z filtru)
    total_lighting = filtered_data["lighting"].sum()
    total_hvac = filtered_data["hvac"].sum()
    total_devices = filtered_data["devices"].sum()

    pie_fig = go.Figure(
        data=[
            go.Pie(
                labels=["Oświetlenie", "HVAC", "Urządzenia"],
                values=[total_lighting, total_hvac, total_devices],
                hole=0.3,
                marker=dict(colors=["#f39c12", "#3498db", "#2ecc71"]),
            )
        ]
    )
    pie_fig.update_layout(
        title="Procentowy udział w zużyciu energii",
        annotations=[dict(text="Energia", x=0.5, y=0.5, font_size=14, showarrow=False)],
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # 3. Histogram - rozkład łącznego zużycia energii
    hist_fig = px.histogram(
        filtered_data,
        x="total_energy",
        nbins=50,
        title="Rozkład łącznego zużycia energii",
        labels={
            "total_energy": "Łączne zużycie energii (kWh)",
            "count": "Liczba wystąpień",
        },
    )
    hist_fig.update_traces(marker_color="#9b59b6")
    hist_fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False
    )
    hist_fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#ecf0f1")
    hist_fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#ecf0f1")

    # 4. Tabela - dzienne zagregowane dane
    filtered_daily = (
        filtered_data.groupby("date")
        .agg(
            {"lighting": "sum", "hvac": "sum", "devices": "sum", "total_energy": "sum"}
        )
        .reset_index()
    )
    filtered_daily.columns = [
        "Data",
        "Oświetlenie (kWh)",
        "HVAC (kWh)",
        "Urządzenia (kWh)",
        "Suma (kWh)",
    ]
    filtered_daily["Data"] = pd.to_datetime(filtered_daily["Data"]).dt.strftime(
        "%Y-%m-%d"
    )

    # Formatowanie wartości w tabeli (zaokrąglenie do 2 miejsc po przecinku)
    for col in ["Oświetlenie (kWh)", "HVAC (kWh)", "Urządzenia (kWh)", "Suma (kWh)"]:
        filtered_daily[col] = filtered_daily[col].round(2)

    # Tekst wyświetlający wybrany zakres
    hover_date = f"{start_date.strftime('%Y-%m-%d')} - {end_date.strftime('%Y-%m-%d')}"

    return line_fig, pie_fig, hist_fig, filtered_daily.to_dict("records"), hover_date


# Uruchomienie aplikacji
if __name__ == "__main__":
    app.run(debug=True, port=8053)
