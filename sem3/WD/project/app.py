import dash
from dash import dcc, html, dash_table
from dash import Input, Output, State, no_update
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from data_preprocessing import (
    load_all_data,
    get_top_scorers,
    get_season_comparison,
    validate_season_for_stats,
    get_available_seasons_for_stat
)

# Initialize app with dark theme
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.CYBORG, '/assets/custom.css'],
                suppress_callback_exceptions=True)

# App Layout
app.layout = html.Div([
    dcc.Store(id='data-store'),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1('NBA History & Evolution Dashboard',
                   className='text-center mb-4',
                   style={'color': '#C9082A', 'font-weight': 'bold'})
        ], width=12)
    ], className='mb-4'),
    
    # Tabs
    dcc.Tabs(id='main-tabs', value='season-explorer', children=[
        dcc.Tab(label='Season Explorer', value='season-explorer'),
        dcc.Tab(label='Head-to-Head', value='head-to-head')
    ], className='mb-4'),
    
    # Loading overlay
    dcc.Loading(
        id='loading-overlay',
        type='default',
        children=[
            html.Div(id='tab-content')
        ]
    ),
    
    # Loading message
    html.Div(id='loading-message', 
             children='Loading data... This may take a few minutes.',
             style={
                 'text-align': 'center',
                 'padding': '50px',
                 'font-size': '18px',
                 'color': 'white',
                 'display': 'block'
             })
])


# Callback: Load data on startup
@app.callback(
    [Output('data-store', 'data'),
     Output('loading-message', 'style')],
    Input('main-tabs', 'value')
)
def load_data(_):
    # Load and process all data
    data = load_all_data()
    
    # Convert DataFrames to dict for storage
    # Note: We don't store 'merged' (1.27 GB) - it's reloaded on-demand
    processed_data = {
        'seasonal_agg': data['seasonal_agg'].to_dict('records'),
        'country_agg': data['country_agg'].to_dict('records'),
        'team_agg': data['team_agg'].to_dict('records'),
        'available_seasons': data['available_seasons']
    }
    
    # Hide loading message
    loading_style = {'display': 'none'}
    
    return processed_data, loading_style


# Callback: Render tab content
@app.callback(
    Output('tab-content', 'children'),
    Input('main-tabs', 'value'),
    State('data-store', 'data')
)
def render_tab(tab, data):
    if data is None:
        return html.Div()
    
    if tab == 'season-explorer':
        return create_season_explorer_layout(data['available_seasons'])
    elif tab == 'head-to-head':
        return create_head_to_head_layout(data['available_seasons'])
    
    return html.Div()


def create_season_explorer_layout(available_seasons):
    """Create layout for Season Explorer tab"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Select Season:', className='text-light mb-2'),
                dcc.Dropdown(
                    id='season-dropdown',
                    options=[{'label': s, 'value': s} for s in available_seasons],
                    value=available_seasons[-1],
                    clearable=False
                )
            ], width=3),
            dbc.Col([
                html.Label('Actions:', className='text-light mb-2'),
                html.Div([
                    dbc.Button(
                        id='apply-explorer-button',
                        children='Apply',
                        color='primary',
                        size='lg',
                        className='w-100',
                        style={'min-width': '120px'}
                    ),
                    html.Div(id='explorer-loading-status',
                             style={'display': 'none', 'margin-top': '10px'})
                ])
            ], width=2)
        ], className='mb-4'),

        dbc.Row([
            dbc.Col([
                dcc.Graph(id='country-map', style={'height': '700px'})
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='top-teams-chart', style={'height': '400px'})
            ], width=6),
            dbc.Col([
                html.H5('Top 10 Scorers', className='text-light mb-3'),
                dash_table.DataTable(
                    id='top-scorers-table',
                    columns=[
                        {'name': 'Rank', 'id': 'rank'},
                        {'name': 'Player', 'id': 'player'},
                        {'name': 'Points', 'id': 'points'}
                    ],
                    page_size=10,
                    style_header={
                        'backgroundColor': '#17408B',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'backgroundColor': '#2C3E50',
                        'color': 'white',
                        'textAlign': 'left'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 0},
                            'backgroundColor': '#C9082A',
                            'color': 'white'
                        },
                        {
                            'if': {'row_index': 1},
                            'backgroundColor': '#17408B',
                            'color': 'white'
                        },
                        {
                            'if': {'row_index': 2},
                            'backgroundColor': '#C9082A',
                            'color': 'white'
                        }
                    ]
                )
            ], width=6)
        ])
    ])


def create_head_to_head_layout(available_seasons):
    """Create layout for Head-to-Head tab"""
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Label('Season A:', className='text-light mb-2'),
                dcc.Dropdown(
                    id='season-a-dropdown',
                    options=[{'label': s, 'value': s} for s in available_seasons],
                    value=available_seasons[-2],
                    clearable=False
                )
            ], width=3),
            dbc.Col([
                html.Label('Season B:', className='text-light mb-2'),
                dcc.Dropdown(
                    id='season-b-dropdown',
                    options=[{'label': s, 'value': s} for s in available_seasons],
                    value=available_seasons[-1],
                    clearable=False
                )
            ], width=3),
            dbc.Col([
                html.Label('Actions:', className='text-light mb-2'),
                html.Div([
                    dbc.Button(
                        id='apply-comparison-button',
                        children='Apply',
                        color='primary',
                        size='lg',
                        className='w-100',
                        style={'min-width': '120px'}
                    ),
                    html.Div(id='comparison-loading-status',
                             style={'display': 'none', 'margin-top': '10px'})
                ])
            ], width=2)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4('3-Pointer Evolution', className='card-title'),
                        html.H2(id='three-pointer-kpi', className='text-primary', style={'font-size': '3rem'})
                    ])
                ], style={'border-color': '#17408B'})
            ], width=12)
        ], className='mb-4'),
        
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='radar-chart', style={'height': '500px'})
            ], width=6),
            dbc.Col([
                dcc.Graph(id='shooting-chart', style={'height': '500px'})
            ], width=6)
        ])
    ])



# Callback: Update country map
@app.callback(
    Output('country-map', 'figure'),
    Output('explorer-loading-status', 'children'),
    Input('apply-explorer-button', 'n_clicks'),
    State('season-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_map(n_clicks, season, data):
    if n_clicks is None or data is None:
        return no_update, no_update

    loading_text = html.Div('Updating...', className='loading-text')

    country_df = pd.DataFrame(data['country_agg'])
    season_data = country_df[country_df['season'] == season]

    if len(season_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this season",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig, loading_text
    
    fig = px.choropleth(
        season_data,
        locations='country',
        locationmode='country names',
        color='player_count',
        hover_name='country',
        hover_data={'country': True, 'player_count': True},
        color_continuous_scale='Viridis',
        title=f'NBA Players by Country - {season}',
        labels={'player_count': 'Number of Players'}
    )
    
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True),
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': 'white'},
        title={'font': {'color': 'white'}}
    )
    
    fig.update_geos(bgcolor='#1a1a1a')

    return fig, loading_text


# Callback: Update top teams chart
@app.callback(
    Output('top-teams-chart', 'figure'),
    Output('explorer-loading-status', 'children', allow_duplicate=True),
    Input('apply-explorer-button', 'n_clicks'),
    State('season-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_top_teams(n_clicks, season, data):
    if n_clicks is None or data is None:
        return no_update, no_update

    loading_text = html.Div('Updating...', className='loading-text')

    team_df = pd.DataFrame(data['team_agg'])
    season_data = team_df[team_df['season'] == season]

    # Filter out pre-1979 seasons for 3-pointer stats
    season_year = int(season.split('-')[0])
    if season_year < 1979:
        fig = go.Figure()
        fig.add_annotation(
            text="3-Pointer statistics not available before 1979",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig, loading_text

    if len(season_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for this season",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig, loading_text

    top_5 = season_data.nlargest(5, 'threePointersAttempted')
    fig = px.bar(
        top_5,
        x='teamName',
        y='threePointersAttempted',
        color='threePointersPercentage',
        color_continuous_scale='Reds',
        title=f'Top 5 Teams - 3-Point Attempts - {season}',
        labels={'threePointersAttempted': '3-Point Attempts',
                'threePointersPercentage': '3P%'}
    )
    
    fig.update_layout(
        xaxis_title='Team',
        yaxis_title='3-Point Attempts',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': 'white'},
        title={'font': {'color': 'white'}},
        xaxis={'tickfont': {'color': 'white'}, 'title': {'font': {'color': 'white'}}},
        yaxis={'tickfont': {'color': 'white'}, 'title': {'font': {'color': 'white'}}}
    )

    return fig, loading_text


# Callback: Update top scorers table
@app.callback(
    Output('top-scorers-table', 'data'),
    Input('apply-explorer-button', 'n_clicks'),
    State('season-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_top_scorers(n_clicks, season, data):
    if n_clicks is None or data is None:
        return no_update

    # Reload merged data from disk (not stored due to size)
    full_data = load_all_data()
    merged_df = full_data['merged']
    season_data = merged_df[merged_df['season'] == season]

    if len(season_data) == 0:
        return no_update
    
    top_10 = (season_data.groupby(['firstName', 'lastName', 'personId'])
              .agg({'points': 'sum'})
              .nlargest(10, 'points')
              .reset_index())
    
    top_10['rank'] = range(1, 11)
    top_10['player'] = top_10['firstName'] + ' ' + top_10['lastName']
    
    return top_10[['rank', 'player', 'points']].to_dict('records')


# Callback: Update radar chart
@app.callback(
    Output('radar-chart', 'figure'),
    Output('comparison-loading-status', 'children'),
    Input('apply-comparison-button', 'n_clicks'),
    State('season-a-dropdown', 'value'),
    State('season-b-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_radar(n_clicks, season_a, season_b, data):
    if n_clicks is None or data is None:
        return no_update, no_update

    loading_text = html.Div('Updating...', className='loading-text')

    # Reload merged data from disk (not stored due to size)
    full_data = load_all_data()
    merged_df = full_data['merged']

    stats = ['points', 'assists', 'reboundsTotal', 'steals', 'blocks', 'turnovers']

    season_a_data = merged_df[merged_df['season'] == season_a]
    season_b_data = merged_df[merged_df['season'] == season_b]

    if len(season_a_data) == 0 or len(season_b_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected seasons",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig, loading_text

    # Calculate average stats per player per game using mean()
    season_a_means = [
        season_a_data['points'].mean(),
        season_a_data['assists'].mean(),
        season_a_data['reboundsTotal'].mean(),
        season_a_data['steals'].mean(),
        season_a_data['blocks'].mean(),
        season_a_data['turnovers'].mean()
    ]

    season_b_means = [
        season_b_data['points'].mean(),
        season_b_data['assists'].mean(),
        season_b_data['reboundsTotal'].mean(),
        season_b_data['steals'].mean(),
        season_b_data['blocks'].mean(),
        season_b_data['turnovers'].mean()
    ]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=season_a_means,
        theta=stats,
        fill='toself',
        name=season_a,
        line_color='#17408B'
    ))

    fig.add_trace(go.Scatterpolar(
        r=season_b_means,
        theta=stats,
        fill='toself',
        name=season_b,
        line_color='#C9082A'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                tickfont={'color': 'white'}
            ),
            angularaxis={'tickfont': {'color': 'white'}}
        ),
        showlegend=True,
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': 'white'},
        legend={'font': {'color': 'white'}}
    )

    return fig, loading_text


# Callback: Update shooting comparison chart
@app.callback(
    Output('shooting-chart', 'figure'),
    Output('comparison-loading-status', 'children', allow_duplicate=True),
    Input('apply-comparison-button', 'n_clicks'),
    State('season-a-dropdown', 'value'),
    State('season-b-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_shooting(n_clicks, season_a, season_b, data):
    if n_clicks is None or data is None:
        return no_update, no_update

    loading_text = html.Div('Updating...', className='loading-text')

    # Reload merged data from disk (not stored due to size)
    full_data = load_all_data()
    merged_df = full_data['merged']

    season_a_data = merged_df[merged_df['season'] == season_a]
    season_b_data = merged_df[merged_df['season'] == season_b]

    if len(season_a_data) == 0 or len(season_b_data) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for selected seasons",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig, loading_text

    # Calculate shooting percentages using weighted averages (sum of makes / sum of attempts)
    # Season A
    season_a_fg_attempts = season_a_data['fieldGoalsAttempted'].sum()
    season_a_3p_attempts = season_a_data['threePointersAttempted'].sum()
    season_a_ft_attempts = season_a_data['freeThrowsAttempted'].sum()

    season_a_fgpct = (season_a_data['fieldGoalsMade'].sum() / season_a_fg_attempts * 100) if season_a_fg_attempts > 0 else 0
    season_a_3ppct = (season_a_data['threePointersMade'].sum() / season_a_3p_attempts * 100) if season_a_3p_attempts > 0 else 0
    season_a_ftpct = (season_a_data['freeThrowsMade'].sum() / season_a_ft_attempts * 100) if season_a_ft_attempts > 0 else 0

    # Season B
    season_b_fg_attempts = season_b_data['fieldGoalsAttempted'].sum()
    season_b_3p_attempts = season_b_data['threePointersAttempted'].sum()
    season_b_ft_attempts = season_b_data['freeThrowsAttempted'].sum()

    season_b_fgpct = (season_b_data['fieldGoalsMade'].sum() / season_b_fg_attempts * 100) if season_b_fg_attempts > 0 else 0
    season_b_3ppct = (season_b_data['threePointersMade'].sum() / season_b_3p_attempts * 100) if season_b_3p_attempts > 0 else 0
    season_b_ftpct = (season_b_data['freeThrowsMade'].sum() / season_b_ft_attempts * 100) if season_b_ft_attempts > 0 else 0

    season_a_shooting = [season_a_fgpct, season_a_3ppct, season_a_ftpct]
    season_b_shooting = [season_b_fgpct, season_b_3ppct, season_b_ftpct]
    shooting_stats = ['Field Goals %', 'Three Pointers %', 'Free Throws %']

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name=season_a,
        x=shooting_stats,
        y=season_a_shooting,
        marker_color='#17408B'
    ))

    fig.add_trace(go.Bar(
        name=season_b,
        x=shooting_stats,
        y=season_b_shooting,
        marker_color='#C9082A'
    ))

    fig.update_layout(
        barmode='group',
        xaxis_title='Shooting Percentage',
        yaxis_title='Percentage (%)',
        paper_bgcolor='#1a1a1a',
        plot_bgcolor='#1a1a1a',
        font={'color': 'white'},
        title={'font': {'color': 'white'}},
        xaxis={'tickfont': {'color': 'white'}, 'title': {'font': {'color': 'white'}}},
        yaxis={'tickfont': {'color': 'white'}, 'title': {'font': {'color': 'white'}}},
        legend={'font': {'color': 'white'}}
    )

    return fig, loading_text


@app.callback(
    Output('three-pointer-kpi', 'children'),
    Input('apply-comparison-button', 'n_clicks'),
    State('season-a-dropdown', 'value'),
    State('season-b-dropdown', 'value'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_three_pointer_kpi(n_clicks, season_a, season_b, data):
    if n_clicks is None or data is None:
        return no_update

    # Reload merged data from disk (not stored due to size)
    full_data = load_all_data()
    merged_df = full_data['merged']

    season_a_data = merged_df[merged_df['season'] == season_a]
    season_b_data = merged_df[merged_df['season'] == season_b]

    # Calculate game counts for each season
    games_a = season_a_data['gameId'].nunique()
    games_b = season_b_data['gameId'].nunique()

    # Handle edge case: no games in season
    if games_a == 0 or games_b == 0:
        return 'N/A'

    # Calculate total 3-point attempts per season
    three_pa_a = season_a_data['threePointersAttempted'].sum()
    three_pa_b = season_b_data['threePointersAttempted'].sum()

    # Calculate 3PA per game
    three_pa_per_game_a = three_pa_a / games_a
    three_pa_per_game_b = three_pa_b / games_b

    # Handle edge case: no 3PA in season A
    if three_pa_per_game_a == 0:
        return 'N/A'

    # Calculate absolute and percentage change
    abs_diff = three_pa_per_game_b - three_pa_per_game_a
    diff_pct = (abs_diff / three_pa_per_game_a) * 100

    symbol_abs = '+' if abs_diff > 0 else ''
    symbol_pct = '+' if diff_pct > 0 else ''

    return f'{symbol_abs}{abs_diff:.2f} attempts/game ({symbol_pct}{diff_pct:.2f}%)'




if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
