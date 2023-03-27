import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from sklearn import metrics
import warnings
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
from streamlit_option_menu import option_menu
import source.title_1 as head
def classification():
    head.title()
    st.markdown("<p style='text-align: center; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement: </span>Application to Classification the Data    </p>", unsafe_allow_html=True)
    st.markdown("<hr style=height:2.5px;background-color:gray>",unsafe_allow_html=True)
    w1,col1,col2,w2,w3=st.columns([1.5,4,4.75,0.25,1.75])
    w11,col11,col22,w22,w33=st.columns([1.5,4,4.75,0.25,1.75])
    w111,col111,col222,w222,w333=st.columns([1.5,4,4.75,0.25,1.75])
    w1111,col1111,col2222,w2222,w3333=st.columns([1.5,4,4.75,0.25,1.75])
    with col1:
        st.write("# ")
        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Problem Statement </span></p>", unsafe_allow_html=True)
    with col2:
        vAR_problem = st.selectbox("",["Select","Student grade classification","Potability of water classification"])
        if vAR_problem == "Student grade classification":
            with col1:
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
            with col2:
                vAR_Model = st.selectbox("",["Select","Decision Tree Classifier"])
            if vAR_Model == "Decision Tree Classifier":
                with col1:
                    st.write("# ")
                    st.write("# ")
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
                        preview_data_1 = vAR_simple_train_file_data.drop('pass',axis = 1)
                        vAR_data_colunms = preview_data_1.columns
                        st.multiselect("",vAR_data_colunms)
                    with col222:      
                        vAR_perameter = st.slider('', 0)
                    with w3:
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        st.write("# ")
                        vAR_preview = st.button("Preview",key=1111)
                    if vAR_preview == True:
                        previwe_data = vAR_simple_train_file_data
                        st.table(previwe_data.head(15))
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
                            #Model building
                            X = data.drop(['pass'],axis=1)
                            y = data['pass']
                            # # splitting X and y into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                            # Create a Decision Tree Classifier object
                            clf = DecisionTreeClassifier()
                            # Create and train the model
                            clf.fit(X, y)
                            st.success("Model trained successfully")
                with col1111:
                    st.write("# ")
                    st.write("# ")
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
                            # Load the testing dataset
                            test_data = pd.read_csv(vAR_simple_test_file)
                            data = vAR_simple_train_file_data
                            #Model building
                            X = data.drop(['pass'],axis=1)
                            y = data['pass']
                            # # splitting X and y into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                            # Create a Decision Tree Classifier object
                            clf = DecisionTreeClassifier()
                            # Create and train the model
                            clf.fit(X, y)
                            y=[]
                            for i,j in test_data.iterrows():
                                x=list(j.values)
                                y.append(x)
                            resfin = []
                            for i in range(0,len(y)):
                                result = clf.predict([y[i]])
                                newres = result[0]
                                resfin.append(newres)
                            test_data['result'] = resfin
                            st.table(test_data)
                                    
        elif vAR_problem == "Potability of water classification":
            with col1:
                st.write("# ")
                st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Model Selection </span></p>", unsafe_allow_html=True)
            with col2:
                vAR_Model = st.selectbox("",["Select","Random Forest classifier"])
            if vAR_Model == "Random Forest classifier":
                with col1:
                    st.write("# ")
                    st.write("### ")
                    st.write("### ")
                    st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Upload Train Data </span></p>", unsafe_allow_html=True)
                with col2:
                    vAR_simple_train_file = st.file_uploader("",type="csv",key='Train')
                if vAR_simple_train_file is not None:
                    vAR_simple_train_file_data =pd.read_csv(vAR_simple_train_file)
                    with col11:
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Feature Selection</span></p>", unsafe_allow_html=True) 
                    with col111:  
                        st.write("# ")
                        st.markdown("<p style='text-align: left; color: black; font-size:20px;'><span style='font-weight: bold'>Hyperparameter</span></p>", unsafe_allow_html=True)
                    with col22:
                        preview_data_1 = vAR_simple_train_file_data.drop('Potability',axis = 1)
                        vAR_data_colunms = preview_data_1.columns
                        st.multiselect("",vAR_data_colunms)
                    with col222:      
                        vAR_perameter = st.slider('', 0)
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
                        previwe_data = vAR_simple_train_file_data
                        st.table(previwe_data.head(15))

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
                            data=data.dropna()
                            #Model building
                            X = data.drop('Potability', axis=1)
                            y = data['Potability']
                            # # splitting X and y into training and testing sets
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
                            # Create a Decision Tree Classifier object
                            rfc = RandomForestClassifier()
                            # Create and train the model
                            rfc.fit(X, y)
                            st.success("Model trained successfully")
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
                                # Load the testing dataset
                                test_data = pd.read_csv(vAR_simple_test_file)
                                test_data.fillna(test_data.mean())
                                data = vAR_simple_train_file_data
                                data=data.dropna(axis=1)
                                #Model building
                                X = data.drop('Potability', axis=1)
                                y = data['Potability']
                                # Create a Random Forest Classifier object
                                rfc = RandomForestClassifier()
                                # Create and train the model
                                rfc.fit(X, y)
                                y=[]
                                for i,j in test_data.iterrows():
                                    x=list(j.values)
                                    y.append(x)
                                resfin = []
                                for i in range(0,len(y)):
                                    result = rfc.predict([y[i]])
                                    newres = result[0]
                                    resfin.append(newres)
                                test_data['Potability'] = resfin
                                st.table(test_data)
    