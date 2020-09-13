import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.dependencies import Input, Output

from build_features import feature_build
from get_data import refresh_data
from process_data import process_data
from sir_simulation import SIR_model

df_input_large = pd.read_csv('data/processed/COVID_final_set.csv', sep=';', parse_dates=["date"])
latest_date = df_input_large["date"].max()
total_confirmed = int(df_input_large[df_input_large["date"] == latest_date]["confirmed"].sum())

countries_df = pd.read_csv("data/processed/country_codes.csv", index_col=None)

for i, row in countries_df.iterrows():
    filt = df_input_large["country"] == row["country"]
    df_input_large.loc[filt, "iso_alpha"] = row["alpha_3"]

df_small = df_input_large[df_input_large["date"] == df_input_large["date"].max()]

pop_df = pd.read_csv("data/processed/populations_2019.csv", index_col="country")

fig = go.Figure()
fig2 = px.scatter_geo(df_small,
                      locations="iso_alpha", size="confirmed",
                      hover_name="country",
                      hover_data={"iso_alpha": False},
                      size_max=40,
                      projection="natural earth",
                      )

app = dash.Dash(__name__)
app.layout = html.Div([

    html.H1(f"CORONAVIRUS DASHBOARD", style={"color": "blue", "text-align": "center"}),
    html.Div(
        [
            html.H3(f"Total confirmed cases as on {latest_date}"),
            html.Span(f"{total_confirmed}",
                      style={"background-color": "yellow", "font-weight": "bold", "font-size": "36px"}),
        ]
    ),

    dcc.Graph(figure=fig2, id="map_scatterplot"),
    dcc.Markdown('''
    ### Country
    ''', style={"color": "blue"}),

    dcc.Dropdown(
        id='country_drop_down',
        options=[{'label': each, 'value': each} for each in df_input_large['country'].unique()],
        value=['US', 'Germany', 'India'],  # which are pre-selected
        multi=True,
        style={"width": "50%"}
    ),

    dcc.Markdown('''
        ### Select Timeline of confirmed COVID-19 cases or the approximated doubling time
        ''', style={"color": "blue"}),

    dcc.Dropdown(
        id='doubling_time',
        options=[
            {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
            {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
            {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
            {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
        ],
        value='confirmed',
        multi=False,
        style={"width": "50%"}
    ),
    html.Br(),

    html.Br(),
    dcc.Graph(figure=fig, id='main_window_slope'),

    html.Br(),

    # SIR Simulation
    html.H2("SIR simulation", style={"color": "blue", "text-align": "center"}),
    dcc.Markdown('''
        ### Country
        ''', style={"color": "blue"}),
    dcc.Dropdown(
        id='country_drop_down2',
        options=[{'label': each, 'value': each} for each in list(pop_df.index)],
        value=['Germany', 'India'],  # which are pre-selected
        multi=True,
        style={"width": "50%"}
    ),

    html.Div(
        [
            html.Div(
                [
                    dcc.Markdown("Initial days without measures", id="initial_days"),
                    dcc.RangeSlider("t_initial", min=1, max=100, step=1, value=[28]),
                    dcc.Markdown("Introducing measures after", id="intro_measures"),
                    dcc.RangeSlider("t_intro_measures", min=1, max=100, step=1, value=[14]),
                    dcc.Markdown("Infection rate", id="beta"),
                    dcc.RangeSlider("infection_rate", min=0, max=1, step=0.01, value=[0.11, 0.4]),
                ],
                style={"width": "50%"}
            ),
            html.Div(
                [
                    dcc.Markdown("Holding measures", id="hold_measures"),
                    dcc.RangeSlider("t_hold", min=1, max=100, step=1, value=[21]),
                    dcc.Markdown("Relaxing measures", id="relax_measures"),
                    dcc.RangeSlider("t_relax", min=1, max=100, step=1, value=[21]),
                    dcc.Markdown("Recovery rate", id="gamma"),
                    dcc.RangeSlider("recovery_rate", min=0, max=1, step=0.1, value=[0.1]),
                ],
                style={"width": "50%"}
            ),
        ],
        style={"display": "flex"}
    ),
    dcc.Graph(id="sir_graph"),

    html.Button("Update data",
                id="refresh_data",
                n_clicks=0,
                style={"background-color": "green", "color": "white", "font-size": "16px"}),
    dcc.Loading(
        id="loading-2",
        children=[html.Div(id="refreshed", style={"display": "inline"})],
        type="circle",
    ),

])


@app.callback(
    Output('main_window_slope', 'figure'),
    [
        Input('country_drop_down', 'value'),
        Input('doubling_time', 'value'),
        Input('refreshed', 'children')
    ])
def update_figure(country_list, show_doubling, refreshed):
    if show_doubling in ['confirmed_DR', 'confirmed_filtered_DR']:
        my_yaxis = {'type': "log",
                    'title': 'Approximated doubling rate over 3 days (larger numbers are better #stayathome)'
                    }
    else:
        my_yaxis = {'type': "log",
                    'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                    }

    traces = []
    for each in country_list:

        df_plot = df_input_large[df_input_large['country'] == each]

        if show_doubling in ['confirmed_DR', 'confirmed_filtered_DR']:
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
        else:
            df_plot = df_plot[
                ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                 'date']].groupby(['country', 'date']).agg(np.sum).reset_index()

        traces.append(dict(x=df_plot.date,
                           y=df_plot[show_doubling],
                           mode='markers+lines',
                           opacity=0.9,
                           name=each
                           )
                      )

    return {
        'data': traces,
        'layout': dict(
            width=1280,
            height=720,
            xaxis={'title': 'Timeline',
                   'tickangle': -45,
                   'nticks': 20,
                   'tickfont': dict(size=14, color="#7f7f7f"),
                   },
            xaxis_rangeslider_visible=True,
            yaxis=my_yaxis
        )
    }


@app.callback(
    [
        Output("sir_graph", "figure"),
        Output("initial_days", "children"),
        Output("intro_measures", "children"),
        Output("hold_measures", "children"),
        Output("relax_measures", "children"),
        Output("beta", "children"),
        Output("gamma", "children"),
    ],
    [
        Input("country_drop_down2", "value"),
        Input("t_initial", "value"),
        Input("t_intro_measures", "value"),
        Input("t_hold", "value"),
        Input("t_relax", "value"),
        Input("infection_rate", "value"),
        Input("recovery_rate", "value"),
    ]
)
def sir_simulation(countries, t_initial, t_intro_measures, t_hold, t_relax, beta, gamma):
    t_initial, t_intro_measures, t_hold, t_relax = [x[0] for x in [t_initial, t_intro_measures, t_hold, t_relax]]

    total = t_initial + t_intro_measures + t_hold + t_relax
    t_phases = np.array([t_initial, t_intro_measures, t_hold, t_relax]).cumsum()

    # beta_max = 0.4
    # beta_min = 0.11
    beta_min, beta_max = beta
    # gamma = 0.1
    gamma = gamma[0]
    pd_beta = np.concatenate((np.array(t_initial * [beta_max]),
                              np.linspace(beta_max, beta_min, t_intro_measures),
                              np.array(t_hold * [beta_min]),
                              np.linspace(beta_min, beta_max, t_relax),
                              ))

    fig = go.Figure()
    for country in countries:
        df_sir = df_input_large.loc[df_input_large.country == country]

        ydata = np.array(df_sir.confirmed[t_initial:total])
        N0 = pop_df.loc[country, "population"]
        I0 = df_sir.confirmed.to_numpy().nonzero()[0][0]
        S0 = N0 - I0
        R0 = 0
        SIR = np.array([S0, I0, R0])
        propagation_rates = pd.DataFrame(columns={'susceptible': S0,
                                                  'infected': I0,
                                                  'recoverd': R0})

        for each_beta in pd_beta:
            new_delta_vec = SIR_model(SIR, each_beta, gamma, N0)

            SIR = SIR + new_delta_vec

            propagation_rates = propagation_rates.append({'susceptible': SIR[0],
                                                          'infected': SIR[1],
                                                          'recovered': SIR[2]}, ignore_index=True)

        fig.add_trace(go.Scatter(x=propagation_rates.index, y=propagation_rates.infected, name=country))
        # fig.add_trace(go.Bar(x=np.arange(len(ydata)), y=ydata, name=country))

    # Highlighting regions with rectangle shapes

    shapes = []
    xs = [0] + [i for i in t_phases]
    opacity = [0.2, 0.3, 0.4, 0.5, 0.6]
    for i in range(4):
        shapes.append(dict(
            type="rect",
            # x-reference is assigned to the x-values
            xref="x",
            # y-reference is assigned to the plot paper [0,1]
            yref="paper",
            x0=xs[i],
            y0=0,
            x1=xs[i + 1],
            y1=1,
            fillcolor="LightSalmon",
            opacity=opacity[i],
            layer="below",
            line_width=0,
        ))

    fig.update_layout(
        width=1280,
        height=720,
        yaxis_type="log",
        shapes=shapes,
        title="SIR simulation",
        xaxis_title="Time in days",
        yaxis_title="Increase in infected cases")

    fig.add_trace(go.Scatter(
        x=list(map(lambda i: i + 5, xs)),
        y=[20] * 4,
        text=["Without measures",
              "Hard measures introduced",
              "Measures held",
              "Measures relaxed"
              ],
        mode="text",
        showlegend=False
    ))

    return fig, f"Initial days without measures = {t_initial}", \
           f"Introduce measures for {t_intro_measures} days", \
           f"Hold the measures for {t_hold} days", \
           f"Relax the measures after {t_relax} days", \
           f"Infection rate [min, max] = [{beta}]", f"Recovery rate = {gamma}"


@app.callback(
    output=Output("refreshed", "children"),
    inputs=[Input("refresh_data", "n_clicks")]
)
def update_data(clicks):
    if clicks > 0:
        refresh_data()
        process_data()
        feature_build()

        return f"Data updated"


if __name__ == '__main__':
    app.run_server(port=8051, debug=True)
