import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request, redirect, flash
import os
import TimeSeriesMLAlgorithm
from OwidWEIBAlgorithm import makePrediction as weibPredict


UPLOAD_FOLDER = os.getcwd()
ALLOWED_EXTENSIONS = set(['csv', 'xlsx'])

hosp_df = pd.read_excel('data/hospitals.xlsx')
state_list = list(hosp_df.State.unique())

server = Flask(__name__, static_url_path='/assets',
               static_folder='./templates/assets',
               template_folder='./templates')
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
server.secret_key = "CovidForeCasting"

app = dash.Dash(__name__,
                server=server)

app.config.suppress_callback_exceptions = True
app.title = 'COVID-19'

dash_colors = {
    'background': '#111111',
    'text': '#BEBEBE',
    'grid': '#333333',
    'red': '#BF0000',
    'blue': '#466fc2',
    'green': '#5bc246'
}

df_worldwide = pd.read_csv('data/df_worldwide.csv')
df_worldwide['percentage'] = df_worldwide['percentage'].astype(str)
df_worldwide['date'] = pd.to_datetime(df_worldwide['date'])

# selects the "data last updated" date
update = df_worldwide['date'].dt.strftime('%B %d, %Y').iloc[-1]

available_countries = sorted(df_worldwide['Country/Region'].unique())

states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
          'Colorado', 'Connecticut', 'Delaware', 'District of Columbia',
          'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
          'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
          'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi',
          'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
          'New Jersey', 'New Mexico', 'New York', 'North Carolina',
          'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania',
          'Rhode Island', 'South Carolina', 'South Dakota', 'Tennessee',
          'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
          'West Virginia', 'Wisconsin', 'Wyoming']

eu = ['Albania', 'Andorra', 'Austria', 'Belarus', 'Belgium',
      'Bosnia and Herzegovina', 'Bulgaria', 'Croatia', 'Cyprus',
      'Czech Republic', 'Denmark', 'Estonia', 'Finland', 'France',
      'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy',
      'Kosovo', 'Latvia', 'Liechtenstein', 'Lithuania', 'Luxembourg',
      'Malta', 'Moldova', 'Monaco', 'Montenegro', 'Netherlands',
      'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania',
      'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden',
      'Switzerland', 'Turkey', 'Ukraine', 'United Kingdom',
      'Vatican City']

china = ['Anhui', 'Beijing', 'Chongqing', 'Fujian', 'Gansu', 'Guangdong',
         'Guangxi', 'Guizhou', 'Hainan', 'Hebei', 'Heilongjiang', 'Henan',
         'Hong Kong', 'Hubei', 'Hunan', 'Inner Mongolia', 'Jiangsu',
         'Jiangxi', 'Jilin', 'Liaoning', 'Macau', 'Ningxia', 'Qinghai',
         'Shaanxi', 'Shandong', 'Shanghai', 'Shanxi', 'Sichuan', 'Tianjin',
         'Tibet', 'Xinjiang', 'Yunnan', 'Zhejiang']

region_options = {'Worldwide': available_countries,
                  'United States': states,
                  'Europe': eu,
                  'China': china}

df_us = pd.read_csv('data/df_us.csv')
df_us['percentage'] = df_us['percentage'].astype(str)
df_us['date'] = pd.to_datetime(df_us['date'])

df_eu = pd.read_csv('data/df_eu.csv')
df_eu['percentage'] = df_eu['percentage'].astype(str)
df_eu['date'] = pd.to_datetime(df_eu['date'])

df_china = pd.read_csv('data/df_china.csv')
df_china['percentage'] = df_china['percentage'].astype(str)
df_china['date'] = pd.to_datetime(df_china['date'])

df_us_counties = pd.concat([pd.read_csv('data/df_us_county1.csv'),
                            pd.read_csv('data/df_us_county2.csv'),
                            pd.read_csv('data/df_us_county3.csv'),
                            pd.read_csv('data/df_us_county4.csv')], ignore_index=True)
df_us_counties['percentage'] = df_us_counties['percentage'].astype(str)
df_us_counties['Country/Region'] = df_us_counties['Country/Region'].astype(str)
df_us_counties['date'] = pd.to_datetime(df_us_counties['date'])


@app.callback(
    Output('confirmed_ind', 'figure'),
    [Input('global_format', 'value')])
def confirmed(view):
    '''
    creates the CUMULATIVE CONFIRMED indicator
    '''
    if view == 'Worldwide':
        df = df_worldwide
    elif view == 'United States':
        df = df_us
    elif view == 'Europe':
        df = df_eu
    elif view == 'China':
        df = df_china
    else:
        df = df_worldwide

    value = df[df['date'] == df['date'].iloc[-1]]['Confirmed'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Confirmed'].sum()
    return {
        'data': [{'type': 'indicator',
                  'mode': 'number+delta',
                  'value': value,
                  'delta': {'reference': delta,
                            'valueformat': ',g',
                            'relative': False,
                            'increasing': {'color': dash_colors['blue']},
                            'decreasing': {'color': dash_colors['green']},
                            'font': {'size': 25}},
                  'number': {'valueformat': ',',
                             'font': {'size': 50}},
                  'domain': {'y': [0, 1], 'x': [0, 1]}}],
        'layout': go.Layout(
            title={'text': "CUMULATIVE CONFIRMED"},
            font=dict(color=dash_colors['red']),
            paper_bgcolor=dash_colors['background'],
            plot_bgcolor=dash_colors['background'],
            height=200
        )
    }


@app.callback(
    Output('active_ind', 'figure'),
    [Input('global_format', 'value')])
def active(view):
    '''
    creates the CURRENTLY ACTIVE indicator
    '''
    if view == 'Worldwide':
        df = df_worldwide
    elif view == 'United States':
        df = df_us
    elif view == 'Europe':
        df = df_eu
    elif view == 'China':
        df = df_china
    else:
        df = df_worldwide

    value = df[df['date'] == df['date'].iloc[-1]]['Active'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Active'].sum()
    return {
        'data': [{'type': 'indicator',
                  'mode': 'number+delta',
                  'value': value,
                  'delta': {'reference': delta,
                            'valueformat': ',g',
                            'relative': False,
                            'increasing': {'color': dash_colors['blue']},
                            'decreasing': {'color': dash_colors['green']},
                            'font': {'size': 25}},
                  'number': {'valueformat': ',',
                             'font': {'size': 50}},
                  'domain': {'y': [0, 1], 'x': [0, 1]}}],
        'layout': go.Layout(
            title={'text': "CURRENTLY ACTIVE"},
            font=dict(color=dash_colors['red']),
            paper_bgcolor=dash_colors['background'],
            plot_bgcolor=dash_colors['background'],
            height=200
        )
    }


@app.callback(
    Output('recovered_ind', 'figure'),
    [Input('global_format', 'value')])
def recovered(view):
    '''
    creates the RECOVERED CASES indicator
    '''
    if view == 'Worldwide':
        df = df_worldwide
    elif view == 'United States':
        df = df_us
    elif view == 'Europe':
        df = df_eu
    elif view == 'China':
        df = df_china
    else:
        df = df_worldwide

    value = df[df['date'] == df['date'].iloc[-1]]['Recovered'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Recovered'].sum()
    return {
        'data': [{'type': 'indicator',
                  'mode': 'number+delta',
                  'value': value,
                  'delta': {'reference': delta,
                            'valueformat': ',g',
                            'relative': False,
                            'increasing': {'color': dash_colors['blue']},
                            'decreasing': {'color': dash_colors['green']},
                            'font': {'size': 25}},
                  'number': {'valueformat': ',',
                             'font': {'size': 50}},
                  'domain': {'y': [0, 1], 'x': [0, 1]}}],
        'layout': go.Layout(
            title={'text': "RECOVERED CASES"},
            font=dict(color=dash_colors['red']),
            paper_bgcolor=dash_colors['background'],
            plot_bgcolor=dash_colors['background'],
            height=200
        )
    }


@app.callback(
    Output('deaths_ind', 'figure'),
    [Input('global_format', 'value')])
def deaths(view):
    '''
    creates the DEATHS TO DATE indicator
    '''
    if view == 'Worldwide':
        df = df_worldwide
    elif view == 'United States':
        df = df_us
    elif view == 'Europe':
        df = df_eu
    elif view == 'China':
        df = df_china
    else:
        df = df_worldwide

    value = df[df['date'] == df['date'].iloc[-1]]['Deaths'].sum()
    delta = df[df['date'] == df['date'].unique()[-2]]['Deaths'].sum()
    return {
        'data': [{'type': 'indicator',
                  'mode': 'number+delta',
                  'value': value,
                  'delta': {'reference': delta,
                            'valueformat': ',g',
                            'relative': False,
                            'increasing': {'color': dash_colors['blue']},
                            'decreasing': {'color': dash_colors['green']},
                            'font': {'size': 25}},
                  'number': {'valueformat': ',',
                             'font': {'size': 50}},
                  'domain': {'y': [0, 1], 'x': [0, 1]}}],
        'layout': go.Layout(
            title={'text': "DEATHS TO DATE"},
            font=dict(color=dash_colors['red']),
            paper_bgcolor=dash_colors['background'],
            plot_bgcolor=dash_colors['background'],
            height=200
        )
    }


@app.callback(
    Output('worldwide_trend', 'figure'),
    [Input('global_format', 'value'),
     Input('population_select', 'value')])
def worldwide_trend(view, population):
    '''
    creates the upper-left chart (aggregated stats for the view)
    '''
    if view == 'Worldwide':
        df = df_worldwide
    elif view == 'United States':
        df = df_us
        df_us.loc[df_us['Country/Region'] == 'Recovered', ['population']] = 0
    elif view == 'Europe':
        df = df_eu
    elif view == 'China':
        df = df_china
    else:
        df = df_worldwide

    if population == 'absolute':
        confirmed = df.groupby('date')['Confirmed'].sum()
        active = df.groupby('date')['Active'].sum()
        recovered = df.groupby('date')['Recovered'].sum()
        deaths = df.groupby('date')['Deaths'].sum()
        title_suffix = ''
        hover = '%{y:,g}'
    elif population == 'percent':
        df = df.dropna(subset=['population'])
        confirmed = df.groupby('date')['Confirmed'].sum() / df.groupby('date')['population'].sum()
        active = df.groupby('date')['Active'].sum() / df.groupby('date')['population'].sum()
        recovered = df.groupby('date')['Recovered'].sum() / df.groupby('date')['population'].sum()
        deaths = df.groupby('date')['Deaths'].sum() / df.groupby('date')['population'].sum()
        title_suffix = ' per 100,000 people'
        hover = '%{y:,.2f}'
    else:
        confirmed = df.groupby('date')['Confirmed'].sum()
        active = df.groupby('date')['Active'].sum()
        recovered = df.groupby('date')['Recovered'].sum()
        deaths = df.groupby('date')['Deaths'].sum()
        title_suffix = ''
        hover = '%{y:,g}'

    traces = [go.Scatter(
        x=df.groupby('date')['date'].first(),
        y=confirmed,
        hovertemplate=hover,
        name="Confirmed",
        mode='lines'),
        go.Scatter(
            x=df.groupby('date')['date'].first(),
            y=active,
            hovertemplate=hover,
            name="Active",
            mode='lines'),
        go.Scatter(
            x=df.groupby('date')['date'].first(),
            y=recovered,
            hovertemplate=hover,
            name="Recovered",
            mode='lines'),
        go.Scatter(
            x=df.groupby('date')['date'].first(),
            y=deaths,
            hovertemplate=hover,
            name="Deaths",
            mode='lines')]
    return {
        'data': traces,
        'layout': go.Layout(
            title="{} Infections{}".format(view, title_suffix),
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            font=dict(color=dash_colors['text']),
            paper_bgcolor=dash_colors['background'],
            plot_bgcolor=dash_colors['background'],
            xaxis=dict(gridcolor=dash_colors['grid']),
            yaxis=dict(gridcolor=dash_colors['grid'])
        )
    }


@app.callback(
    Output('country_select', 'options'),
    [Input('global_format', 'value')])
def set_active_options(selected_view):
    '''
    sets allowable options for regions in the upper-right chart drop-down
    '''
    return [{'label': i, 'value': i} for i in region_options[selected_view]]


@app.callback(
    Output('country_select', 'value'),
    [Input('global_format', 'value'),
     Input('country_select', 'options')])
def set_countries_value(view, available_options):
    '''
    sets default selections for regions in the upper-right chart drop-down
    '''
    if view == 'Worldwide':
        return ['US', 'Italy', 'United Kingdom', 'Spain', 'Russia', 'Brazil', 'Sweden', 'Belgium', 'Peru', 'India',
                'Lithuania']
    elif view == 'United States':
        return ['New York', 'New Jersey', 'California', 'Texas', 'Florida', 'Mississippi', 'Arizona', 'Louisiana',
                'Colorado']
    elif view == 'Europe':
        return ['France', 'Germany', 'Italy', 'Spain', 'United Kingdom', 'Belgium', 'Sweden', 'Lithuania']
    elif view == 'China':
        return ['Hubei', 'Guangdong', 'Xinjiang', 'Zhejiang', 'Hunan', 'Hong Kong', 'Macau']
    else:
        return ['US', 'Italy', 'United Kingdom', 'Spain', 'France', 'Germany', 'Russia']


@app.callback(
    Output('active_countries', 'figure'),
    [Input('global_format', 'value'),
     Input('country_select', 'value'),
     Input('column_select', 'value'),
     Input('population_select', 'value')])
def active_countries(view, countries, column, population):
    '''
    creates the upper-right chart (sub-region analysis)
    '''
    if view == 'Worldwide':
        df = df_worldwide
    elif view == 'United States':
        df = df_us
    elif view == 'Europe':
        df = df_eu
    elif view == 'China':
        df = df_china
    else:
        df = df_worldwide

    if population == 'absolute':
        column_label = column
        hover = '%{y:,g}<br>%{x}'
    elif population == 'percent':
        column_label = '{} per 100,000'.format(column)
        df = df.dropna(subset=['population'])
        hover = '%{y:,.2f}<br>%{x}'
    else:
        column_label = column
        hover = '%{y:,g}<br>%{x}'

    traces = []
    countries = df[(df['Country/Region'].isin(countries)) &
                   (df['date'] == df['date'].max())].groupby('Country/Region')['Confirmed'].sum().sort_values(
        ascending=False).index.to_list()
    for country in countries:
        if population == 'absolute':
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum()
        elif population == 'percent':
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum() / \
                     df[df['Country/Region'] == country].groupby('date')['population'].first()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum() / \
                        df[df['Country/Region'] == country].groupby('date')['population'].first()
        else:
            y_data = df[df['Country/Region'] == country].groupby('date')[column].sum()
            recovered = df[df['Country/Region'] == 'Recovered'].groupby('date')[column].sum()

        traces.append(go.Scatter(
            x=df[df['Country/Region'] == country].groupby('date')['date'].first(),
            y=y_data,
            hovertemplate=hover,
            name=country,
            mode='lines'))
    if column == 'Recovered':
        traces.append(go.Scatter(
            x=df[df['Country/Region'] == 'Recovered'].groupby('date')['date'].first(),
            y=recovered,
            hovertemplate=hover,
            name='Unidentified',
            mode='lines'))
    return {
        'data': traces,
        'layout': go.Layout(
            title="{} by Region".format(column_label),
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            font=dict(color=dash_colors['text']),
            paper_bgcolor=dash_colors['background'],
            plot_bgcolor=dash_colors['background'],
            xaxis=dict(gridcolor=dash_colors['grid']),
            yaxis=dict(gridcolor=dash_colors['grid']),
            hovermode='closest'
        )
    }


def hex_to_rgba(h, alpha=1):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return tuple([int(h.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4)] + [alpha])


app.layout = html.Div(style={'backgroundColor': dash_colors['background']}, children=[
    html.H1(children='COVID-19',
            style={
                'textAlign': 'center',
                'color': dash_colors['text']
            }
            ),

    html.Div(children='Data last updated {} end-of-day'.format(update), style={
        'textAlign': 'center',
        'color': dash_colors['text']
    }),

    html.Div(children='Select focus for the dashboard:', style={
        'textAlign': 'center',
        'color': dash_colors['text']
    }),

    html.Div(dcc.RadioItems(id='global_format',
                            options=[{'label': i, 'value': i} for i in
                                     ['Worldwide', 'United States', 'Europe', 'China']],
                            value='Worldwide',
                            labelStyle={'float': 'center', 'display': 'inline-block'}
                            ), style={'textAlign': 'center',
                                      'color': dash_colors['text'],
                                      'width': '100%',
                                      'float': 'center',
                                      'display': 'inline-block'
                                      }
             ),

    html.Div(dcc.Graph(id='confirmed_ind'),
             style={
                 'textAlign': 'center',
                 'color': dash_colors['red'],
                 'width': '33%',
                 'float': 'left',
                 'display': 'inline-block'
             }
             ),

    html.Div(dcc.Graph(id='active_ind'),
             style={
                 'textAlign': 'center',
                 'color': dash_colors['red'],
                 'width': '33%',
                 'float': 'left',
                 'display': 'inline-block'
             }
             ),

    html.Div(dcc.Graph(id='deaths_ind'),
             style={
                 'textAlign': 'center',
                 'color': dash_colors['red'],
                 'width': '33%',
                 'float': 'left',
                 'display': 'inline-block'
             }
             ),

    html.Div(dcc.Graph(id='recovered_ind'),
             style={
                 'textAlign': 'center',
                 'color': dash_colors['red'],
                 'width': '25%',
                 'float': 'left',
                 'display': 'none'
             }
             ),

    html.Div(dcc.Markdown('Display data in the below two charts as total values or as values relative to population:'),
             style={
                 'textAlign': 'center',
                 'color': dash_colors['text'],
                 'width': '100%',
                 'float': 'center',
                 'display': 'inline-block'}),

    html.Div(dcc.RadioItems(id='population_select',
                            options=[{'label': 'Total values', 'value': 'absolute'},
                                     {'label': 'Values per 100,000 of population', 'value': 'percent'}],
                            value='absolute',
                            labelStyle={'float': 'center', 'display': 'inline-block'},
                            style={'textAlign': 'center',
                                   'color': dash_colors['text'],
                                   'width': '100%',
                                   'float': 'center',
                                   'display': 'inline-block'
                                   })
             ),

    html.Div(  # worldwide_trend and active_countries
        [
            html.Div(
                dcc.Graph(id='worldwide_trend'),
                style={'width': '50%', 'float': 'left', 'display': 'inline-block'}
            ),
            html.Div([
                dcc.Graph(id='active_countries'),
                html.Div([
                    dcc.RadioItems(
                        id='column_select',
                        options=[{'label': i, 'value': i} for i in ['Confirmed', 'Active', 'Recovered', 'Deaths']],
                        value='Confirmed',
                        labelStyle={'float': 'center', 'display': 'inline-block'},
                        style={'textAlign': 'center',
                               'color': dash_colors['text'],
                               'width': '100%',
                               'float': 'center',
                               'display': 'inline-block'
                               }),
                    dcc.Dropdown(
                        id='country_select',
                        multi=True,
                        style={'width': '95%', 'float': 'center'}
                    )],
                    style={'width': '100%', 'float': 'center', 'display': 'inline-block'})
            ],
                style={'width': '50%', 'float': 'right', 'vertical-align': 'bottom'}
            )],
        style={'width': '98%', 'float': 'center', 'vertical-align': 'bottom'}
    )

])




@server.route('/hospital', methods=['POST', 'GET'])
def hospital():
    if request.method == 'POST':
        state = request.form["state"]
        hospitals = list(hosp_df[hosp_df['State']==state]['Name & Address of the Hospital'])
        cities = list(hosp_df[hosp_df['State']==state]['City'])

        print(len(hospitals), len(cities))

        hosp_data = []
        for i in range(len(hospitals)):
            print(i)
            hosp_data.append((i+1, state, cities[i], hospitals[i]))

        return render_template('hospital.html',state_list= state_list, hosp_data=hosp_data, output="show")
    return render_template('hospital.html',state_list= state_list)

@server.route('/index', methods=['POST', 'GET'])
def index():
    return render_template('index.html')

@server.route('/prediction', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        country = request.form["country"]
        days = request.form["days"]
        algo = request.form["algo"]

        days_since_1_22, all_cases, start_date, future_forcast_dates, adjusted_dates, future_forecast, days_in_future = TimeSeriesMLAlgorithm.getData(
            country, int(days))

        result = TimeSeriesMLAlgorithm.makePrediction(
            algo, days_since_1_22, all_cases, start_date, future_forcast_dates, adjusted_dates, future_forecast,
            days_in_future,os.path.join(server.config['UPLOAD_FOLDER']),country
        )

        futureDates = []
        futureCases = []
        data = list(result[1])
        data.sort(key = lambda x: x[1])
        for i in range(len(data)):
            data[i] = (data[i][0],int(data[i][1]))
            futureCases.append(int(data[i][1]))
            futureDates.append(data[i][0])

        return render_template('prediction.html',output="show",future_cases=data)

    else:
        return render_template('prediction.html')


@server.route('/predictionUsingWeib', methods=['POST', 'GET'])
def predictionUsingWeib():
    if request.method == 'POST':
        country = request.form["country1"]
        result = weibPredict(country)

    return render_template('plots/graph.html')

@server.route('/uploaded_chest', methods=['POST', 'GET'])
def uploaded_chest():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(server.config['UPLOAD_FOLDER'], r'timeSeriesDataset\time_series_covid19_confirmed_global.csv'))
            flash('Uploaded Successfully')
            return redirect(request.url)

    return render_template('prediction.html')


if __name__ == '__main__':
    server.run(debug=True)
    # app.run_server(debug=True)
