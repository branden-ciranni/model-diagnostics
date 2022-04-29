from textwrap import wrap
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold


from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import LearningCurve
from streamlit_yellowbrick import st_yellowbrick



import os

DATA_PATH = 'data/'

st.title('Model Diagnostics on Student Alcoholism')



def load_data():
    df = pd.read_csv(os.path.join(DATA_PATH, 'student-mat.csv'))
    df['total_grade'] = df[['G1', 'G2', 'G3']].sum(axis=1)/60
    df['y'] = df.total_grade > 0.65
    return df


if __name__ == '__main__':
    df = load_data()
    categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
               'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
               'higher', 'internet', 'romantic']

    numeric = [col for col in df.iloc[:, :30].columns if col not in categorical]
    cols = categorical + numeric

    selected_cols = st.sidebar.multiselect('Choose Features', options = cols, default = cols)
    lambda_reg = st.sidebar.slider('Select Regularization Lambda', 0.0, 5.0, 1.0, step=0.1)

    if lambda_reg == 0:
        lambda_reg = 10**(-10)

    X = df[selected_cols]
    categorical_selected = [col for col in categorical if col in selected_cols]
    numeric_selected = [col for col in numeric if col in selected_cols]

    if categorical_selected:
        le = OneHotEncoder()
        X_cat = le.fit_transform(X[categorical_selected].values).toarray()
    if numeric_selected:
        X_numeric = X[numeric_selected].values
    
    if categorical_selected and numeric_selected:
        X = np.concatenate([X_cat, X_numeric], axis=1)
    elif categorical_selected:
        X = X_cat
    elif numeric_selected:
        X = X_numeric
    else:
        X = []

    y = df.y

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    clf = LogisticRegression(random_state=0, C=1/lambda_reg, max_iter=1000).fit(X_train,y_train)

    visualizer = ROCAUC(clf, classes=['fail', 'pass'])

    st.subheader('Fitted Model')
    intercept = '%.2f' % clf.intercept_
    coef = clf.coef_[0]
    coef_formatted = ' + '.join([f'%.2f({col})' % beta for beta,col in list(zip(coef, selected_cols))])
    # st.text(coef)
    st.latex(r'\text{logit}(\pi) = ' + r'\\'.join(wrap(f'{intercept} + {coef_formatted}', subsequent_indent='\t')))

    st.subheader('ROC Curves for Model')

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    st_yellowbrick(visualizer)      # Finalize and render the figure


    st.subheader('Learning Curves for Model')

    cv = StratifiedKFold(n_splits=10)
    sizes = np.linspace(0.3, 1.0, 10)

    model = LogisticRegression(random_state=0, C=1/lambda_reg, max_iter=500)

    visualizer = LearningCurve(
    model, cv=cv, train_sizes=sizes, n_jobs=4)

    visualizer.fit(X, y)        # Fit the data to the visualizer
    st_yellowbrick(visualizer)
