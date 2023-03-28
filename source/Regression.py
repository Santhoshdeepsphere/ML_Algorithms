import streamlit as st
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
from streamlit_option_menu import option_menu
import source.title_1 as head
def regression():
    head.title()
    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Regression model</p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    w1,col1,col2,w2,w3=st.columns([1.5,4,4.75,0.25,1.75])
    w11,col11,col22,w22,w33=st.columns([1.5,4,4.75,0.25,1.75])
    w111,col111,col222,w222,w333=st.columns([1.5,4,4.75,0.25,1.75])
    w1111,col1111,col2222,w2222,w3333=st.columns([1.5,4,4.75,0.25,1.75])
    with col1:
        st.write("# ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
    with col2:
        vAR_problem = st.selectbox("",["Select","Predicting Apartment Price","Predicting Mileage"])
        if vAR_problem == "Predicting Apartment Price":
            with col1:
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
            with col2:
                vAR_Model = st.selectbox("",["Select","Simple Linear Regression"])
            if vAR_Model == "Simple Linear Regression":
                with col1:
                    st.write("# ")
                    st.write("### ")
                    st.write("### ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
                vAR_simple_train_file = st.file_uploader("",type="csv",key='Train')
                if vAR_simple_train_file is not None:
                    vAR_simple_train_file_data = pd.read_csv(vAR_simple_train_file)
                    with col11:
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Feature Selection</span></p>", unsafe_allow_html=True) 
                    with col111:  
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Hyperparameter</span></p>", unsafe_allow_html=True)
                    with col22:
                        preview_data_1 = vAR_simple_train_file_data.drop('Selling Price (in $1000s)',axis=1)
                        vAR_data_colunms = preview_data_1.columns
                        vAR_Features = st.multiselect("",vAR_data_colunms)
                    with col222:      
                        vAR_perameter = st.slider('', min_value=0.05, max_value=0.3,step=0.05)
                    with w3:
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        vAR_preview = st.button("Preview")
                    if vAR_preview == True:
                        st.table(vAR_simple_train_file_data.head(15))
                if vAR_simple_train_file is not None:
                    with col222:
                        button_placeholder = st.empty()
                        button_clicked = False
                        key=0
                        while not button_clicked:
                            key=key+1
                            button_clicked = button_placeholder.button('Train',key=key)
                            break
                        if button_clicked:
                            button_placeholder.empty()
                            # Load the training dataset
                            data = vAR_simple_train_file_data
                            X = data[vAR_Features]
                            y = data[['Selling Price (in $1000s)']]
                            # # splitting X and y into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vAR_perameter, random_state=1)
                            # Create a linear regression object
                            lr = LinearRegression()
                            # Create and train the model
                            model = LinearRegression()
                            model.fit(X, y)
                            st.success("Model Training is successfull")
                with col1111:
                    st.write("# ")
                    st.write("### ")
                    st.write("### ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
                    with col2222:
                        vAR_simple_test_file = st.file_uploader("",type="csv",key='Test') 
                        if  vAR_simple_test_file is not None:
                            with col2222:
                                button_placeholder = st.empty()
                                button_clicked = False
                                key=2
                                while not button_clicked:
                                    key=key+2
                                    button_clicked = button_placeholder.button('Test',key=key)
                                    break
                                if button_clicked:
                                    button_placeholder.empty()
                                    # Load the training dataset
                                    data = vAR_simple_train_file_data
                                    vAR_test_data = pd.read_csv(vAR_simple_test_file)
                                    vAR_test_data_columns = list(vAR_test_data.columns)
                                    for i in vAR_test_data_columns:
                                        if i in vAR_Features:
                                            continue
                                        else:
                                            vAR_test_data_columns.remove(i)
                                    vAR_test_final_data =  vAR_test_data[vAR_test_data_columns]
                                    X = data[vAR_Features]
                                    y = data['Selling Price (in $1000s)']

                                    # # splitting X and y into training and testing sets
                                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vAR_perameter, random_state=1)

                                    # Create a linear regression object
                                    lr = LinearRegression()

                                    # Train the model
                                    lr.fit(X_train, y_train)
                                    y=[]
                                    for i,j in vAR_test_final_data.iterrows():
                                        x=list(j.values)
                                        y.append(x)
                                    resfin = []
                                    for i in range(0,len(y)):
                                        result = lr.predict([y[i]])
                                        newres = result[0]
                                        resfin.append(newres)
                                    vAR_test_final_data['Selling Price (in $1000s)'] = resfin
                                    st.table(vAR_test_final_data)
                                
        if vAR_problem == "Predicting Mileage":
            with col1:
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
            vAR_Model = st.selectbox("",["Select","Mulitiple Linear Regression"])
            if vAR_Model == "Mulitiple Linear Regression":
                with col1:
                    st.write("# ")
                    st.write("### ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
                vAR_multi_train_file = st.file_uploader("",type="csv",key='Train')
                if vAR_multi_train_file is not None:
                    vAR_multi_train_file_data = pd.read_csv(vAR_multi_train_file)
                    with col11:
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Feature Selection</span></p>", unsafe_allow_html=True) 
                    with col111:  
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Hyperparameter</span></p>", unsafe_allow_html=True)
                    with col22:
                        preview_data_1 = vAR_multi_train_file_data.drop('Mileage (in mpg)',axis = 1)
                        vAR_data_colunms = preview_data_1.columns
                        vAR_Features = st.multiselect("",vAR_data_colunms)
                    with col222:      
                        vAR_perameter = st.slider('', min_value=0.05, max_value=0.3,step=0.05)
                    with w3:
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        vAR_preview = st.button("Preview")
                    if vAR_preview == True:
                        st.table(vAR_multi_train_file_data.head(15))
                if vAR_multi_train_file is not None:
                    with col222:
                        button_placeholder = st.empty()
                        button_clicked = False
                        key=0
                        while not button_clicked:
                            key=key+1
                            button_clicked = button_placeholder.button('Train',key=key)
                            break
                        if button_clicked:
                            button_placeholder.empty()
                            # Load the training dataset
                            data = vAR_multi_train_file_data

                            X = data[vAR_Features]
                            y = data.iloc[:,-1].values

                            # # splitting X and y into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vAR_perameter, random_state=1)

                            # Create a linear regression object
                            lr = LinearRegression()

                            # Train the model
                            lr.fit(X_train, y_train)
                            st.success("Model Training is successfull")
                    with col1111:
                        st.write("# ")
                        st.write("### ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Test Data </span></p>", unsafe_allow_html=True)
                    with col2222:
                        vAR_simple_test_file = st.file_uploader("",type="csv",key='Test') 
                        if  vAR_simple_test_file is not None:
                            button_placeholder = st.empty()
                            button_clicked = False
                            key=2
                            while not button_clicked:
                                key=key+2
                                button_clicked = button_placeholder.button('Test',key=key)
                                break
                            if button_clicked:
                                button_placeholder.empty()
                                # Load the training dataset
                                data = vAR_multi_train_file_data
                                vAR_test_data = pd.read_csv(vAR_simple_test_file)
                                vAR_test_data_columns = list(vAR_test_data.columns)
                                for i in vAR_test_data_columns:
                                    if i in vAR_Features:
                                        continue
                                    else:
                                        vAR_test_data_columns.remove(i)
                                vAR_test_final_data =  vAR_test_data[vAR_test_data_columns]
                                X = data[vAR_Features]
                                y = data.iloc[:,-1].values

                                # # splitting X and y into training and testing sets
                                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=vAR_perameter, random_state=1)

                                # Create a linear regression object
                                lr = LinearRegression()

                                # Train the model
                                lr.fit(X, y)
                                y=[]
                                for i,j in vAR_test_final_data.iterrows():
                                    x=list(j.values)
                                    y.append(x)
                                resfin = []
                                for i in range(0,len(y)):
                                    result = lr.predict([y[i]])
                                    newres = result[0]
                                    resfin.append(newres)
                                vAR_test_final_data['Mileage (in mpg)'] = resfin
                                st.table(vAR_test_final_data)
