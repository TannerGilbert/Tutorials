import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def main():

    df = load_data()

    page = st.sidebar.selectbox("Choose a page", ['Homepage', 'Exploration', 'Prediction'])

    if page == 'Homepage':
        st.title('Wine Alcohol Class Prediction')
        st.text('Select a page in the sidebar')
        st.dataframe(df)
    elif page == 'Exploration':
        st.title('Explore the Wine Data-set')
        if st.checkbox('Show column descriptions'):
            st.dataframe(df.describe())
        
        st.markdown('### Analysing column relations')
        st.text('Correlations:')
        fig, ax = plt.subplots(figsize=(10,10))
        sns.heatmap(df.corr(), annot=True, ax=ax)
        st.pyplot()
        st.text('Effect of the different classes')
        sns.pairplot(df, vars=['magnesium', 'flavanoids', 'nonflavanoid_phenols', 'proline'], hue='alcohol')
        st.pyplot()
    else:
        st.title('Modelling')
        model, accuracy = train_model(df)
        st.write('Accuracy: ' + str(accuracy))
        st.markdown('### Make prediction')
        st.dataframe(df)
        row_number = st.number_input('Select row', min_value=0, max_value=len(df)-1, value=0)
        st.markdown('#### Predicted')
        st.text(model.predict(df.drop(['alcohol'], axis=1).loc[row_number].values.reshape(1, -1))[0])


@st.cache
def train_model(df):
    X = np.array(df.drop(['alcohol'], axis=1))
    y= np.array(df['alcohol'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, model.score(X_test, y_test)

@st.cache
def load_data():
    return pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', names=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols','flavanoids', 'nonflavanoid_phenols' ,'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'], delimiter=",", index_col=False)


if __name__ == '__main__':
    main()