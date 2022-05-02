from statistics import mean
from textwrap import wrap
from soupsieve import select
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn import metrics


from yellowbrick.classifier import ROCAUC
from yellowbrick.model_selection import LearningCurve
from streamlit_yellowbrick import st_yellowbrick
import seaborn as sns
import matplotlib.pyplot as plt



import os

DATA_PATH = './data/'

st.title('Model Diagnostics on the effects of Student Alcohol Consumption on Grades')

def load_data():
    df = pd.read_csv(os.path.join(DATA_PATH, 'student-mat.csv'))
    df['total_grade'] = df[['G1', 'G2', 'G3']].sum(axis=1)
    df['y'] = df.total_grade >= df.total_grade.median()
    return df

def subset_data(df, selected_cols, categorical, numeric):
    X = df[selected_cols]
    categorical_selected = [col for col in categorical if col in selected_cols]
    numeric_selected = [col for col in numeric if col in selected_cols]
    return X, categorical_selected, numeric_selected

def transform(X, categorical_cols, numeric_cols):
    if categorical_cols:
        oh = OneHotEncoder()
        X_cat = oh.fit_transform(X[categorical_cols].values).toarray()
    if numeric_cols:
        scaler = StandardScaler()
        X_numeric = scaler.fit_transform(X[numeric_cols].values)
    
    if categorical_cols and numeric_cols:
        X = np.concatenate([X_cat, X_numeric], axis=1)
    elif categorical_cols:
        X = X_cat
    elif numeric_cols:
        X = X_numeric
    else:
        X = []

    return X


def deviance(X, y, model):
    return 2*metrics.log_loss(y, model.predict_log_proba(X))
    


if __name__ == '__main__':
    data_dict = pd.read_csv(os.path.join(DATA_PATH, 'data-dictionary.csv'), sep='|')
    st.subheader('Data Dictionary')
    with st.expander('Click to show Column Descriptions'):
        st.dataframe(data_dict)

    df = load_data()

    
    # initial_subset = []


    categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason',
               'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
               'higher', 'internet', 'romantic']

    numeric = [col for col in df.iloc[:, :30].columns if col not in categorical]
    cols = categorical + numeric

    selected_cols = st.sidebar.multiselect('Choose Features', options = cols, default = cols)
    lambda_reg = st.sidebar.slider('Select Regularization Lambda', 0.0, 5.0, 1.0, step=0.1)

    if lambda_reg == 0:
        lambda_reg = 10**(-10)

    X, categorical_selected, numeric_selected = subset_data(df, selected_cols, categorical, numeric)

    X_transform = transform(X, categorical_selected, numeric_selected)
    y = df.y

    X_train, X_test, y_train, y_test = train_test_split(X_transform, y, random_state=42)
    lr = LogisticRegression(random_state=42, C=1/lambda_reg, max_iter=1000)
    clf = lr.fit(X_train,y_train)



    
    visualizer = ROCAUC(clf, per_class=False, binary=True)

    st.subheader('Fitted Model')
    intercept = '%.2f' % clf.intercept_
    coef = clf.coef_[0]
    coef_formatted = ' '.join([f'{"+" if beta > 0 else ""}%.2f({col})' % beta for beta,col in list(zip(coef, selected_cols))])
    # st.text(coef)
    st.latex(r'\text{logit}(\pi) = ' + r'\\'.join(wrap(f'{intercept} {coef_formatted}', subsequent_indent='\t')))

    
    
    
    st.subheader('ROC Curves for Model')

    visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
    visualizer.score(X_test, y_test)        # Evaluate the model on the test data
    st_yellowbrick(visualizer)      # Finalize and render the figure


    
    
    st.subheader('Learning Curves for Model')

    cv = StratifiedKFold(n_splits=10)
    sizes = np.linspace(0.3, 1.0, 10)

    model = LogisticRegression(random_state=42, C=1/lambda_reg, max_iter=500)

    visualizer = LearningCurve(
    model, cv=cv, train_sizes=sizes, n_jobs=4)

    visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
    st_yellowbrick(visualizer)




    st.subheader('Correlation Matrix')

    corrmat = X.corr()
    top_corr_features = corrmat.index
    fig, ax = plt.subplots()
    g = sns.heatmap(X[top_corr_features].corr(), cmap="YlGnBu", ax=ax).figure
    st.pyplot(g)



