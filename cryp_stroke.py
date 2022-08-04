import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.model_selection import cross_val_score
import random
#import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler

#应用主题

#应用标题
st.title('Machine Learning Application for Predicting Cryptogenic stroke')



# conf
col1, col2, col3 = st.columns(3)
RoPE = col1.number_input('Age',step=1,value=4)
SD = col2.selectbox("Stroke distribution",('Anterior circulation','Posterior circulation','Anterior/posterior circulation'))
SOS = col3.selectbox("Side of hemisphere",('Left','Right','Bilateral'))
NOS = col1.selectbox("Site of stroke lesion",('Cortex','Cortex-subcortex','Subcortex','Brainstem','Cerebellum'))
Ddimer = col2.number_input('D-dimer (ng/mL)',value=174)
BNP = col3.number_input('BNP (pg/mL)',value=93)
map = {'Left':0,'Right':1,'Bilateral':2,
       'Anterior circulation':0,'Posterior circulation':1,'Anterior/posterior circulation':2,
       'Cortex':0,'Cortex-subcortex':1,'Subcortex':2,'Brainstem':3,'Cerebellum':4,
       'No':0,'Yes':1}

SD =map[SD]
SOS =map[SOS]
NOS =map[NOS]
# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
# thyroid_train['fracture'] = thyroid_train['fracture'].apply(lambda x : +1 if x==1 else 0)
features=[  'RoPE', 'SD',  'SOS', 'NOS',  'Ddimer', 'BNP']
target='Group'

#处理数据不平衡
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

XGB = XGBClassifier(random_state=32,max_depth=5,n_estimators=32)
XGB.fit(X_ros, y_ros)


sp = 0.5
#figure
is_t = (XGB.predict_proba(np.array([[RoPE, SD,  SOS, NOS,  Ddimer, BNP]]))[0][1])> sp
prob = (XGB.predict_proba(np.array([[RoPE, SD,  SOS, NOS,  Ddimer, BNP]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%'
