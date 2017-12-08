import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
from pandas_datareader import data as web
from datetime import datetime as dt
import plotly.graph_objs as go
import pandas as pd

app = dash.Dash()

from sqlalchemy import create_engine

mysql_engine = create_engine('mysql+mysqlconnector://ca_elo_games:cprice31!@mysql.crowdscoutsports.com:3306/nhl_all')

crowdscout_data = pd.read_sql("SELECT Player, season, `Predicted.CS` as Predicted_CS FROM `nhl_all`.`crowdscout_data_predictions`",con=mysql_engine)
player_scores = crowdscout_data.sort_values(['season'])

player_scores['season'] = player_scores['season'].astype(str).str.slice(4,8)
                                                  
player_list = crowdscout_data_download.Player.drop_duplicates()

app.layout = html.Div([
    html.H1('Player CrowdScout Scores by Season'),
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='p1',
                options=[{'label': i, 'value': i} for i in player_list],
                value='SIDNEY CROSBY'
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='p2',
                options=[{'label': i, 'value': i} for i in player_list],
                value='CONNOR MCDAVID'
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    dcc.Graph(id='indicator-graphic')

  #  dcc.Slider(
  #      id='year',
   #     min=player_scores['season'].min(),
    #    max=player_scores['season'].max(),
     #   value=player_scores['season'].max(),
     #   step=None,
     #   marks={str(season): str(season) for season in player_scores['season'].unique()}
    #)
])

@app.callback(
    dash.dependencies.Output('indicator-graphic', 'figure'),
    [dash.dependencies.Input('p1', 'value'),
     dash.dependencies.Input('p2', 'value')
    # dash.dependencies.Input('year', 'value')
    ])
def update_graph(p1, p2):
    player_df = player_scores.loc[player_scores.Player.isin([p1, p2]),:]
    return {
        'data': [
        {
            'x': player_df[player_df['Player']==player]['season'],
            'y': player_df[player_df['Player']==player]['Predicted_CS'],
            'name': player, 'mode': 'lines',
        } for player in [p1, p2]
    ],
        'layout': go.Layout(
                yaxis={'title': 'Predicted CrowdScout Score',
                      'range':([0,101]) },
                xaxis={'title': 'Season','type':'category'},
            
                )


    }

if __name__ == '__main__':
    app.run_server()