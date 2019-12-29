import streamlit as st
import numpy as np
import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go

st.title('Stock Price of Apple')

@st.cache
def load_data():
    data = pd.read_csv('AAPL_data.csv', parse_dates=['date'])
    return data

df = load_data()

columns = st.multiselect(
    "Choose Columns", list(df.drop(['date', 'Name'], axis=1).columns), ['open']
)

columns.extend(['date'])

start_date = st.date_input('Start date', value=df['date'].min())
end_date = st.date_input('End date', value=df['date'].max())

data = df[columns][(df['date']>=start_date) & (df['date']<=end_date)]

st.write(data)

st.subheader('Line chart of selected columns')
chart = st.line_chart(data.drop(['date'], axis=1))

if st.checkbox('Show summaries'):
    st.subheader('Summaries:')
    st.write(data.describe())

    week_df = data.groupby(data['date'].dt.weekday_name).mean()
    
    traces = [go.Bar(
        x = week_df.index,
        y = data[col],
        name = col,
        marker = dict(
            line = dict(
                color = 'rgb(0, 0, 0)',
                width = 2
            )
        )
    ) for col in data.drop(['date'], axis=1).columns]

    layout = go.Layout(
        title = 'Stockprice over days',
        xaxis = dict(
            title = 'Weekday',
        ),
        yaxis = dict(
            title = 'Average Price'
        )
    )

    fig = go.Figure(data=traces, layout=layout)

    st.plotly_chart(fig)