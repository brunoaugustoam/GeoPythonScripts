import numpy as np

import matplotlib.pyplot as plt
import mplstereonet as mpl
import pandas as pd
from utils import *
from plots import *
import streamlit as st



st.set_page_config(layout='wide')
np.random.seed(0)

#Future updates should give the user more control over the inputed data, distribution of the data, and the number of structures to be generated
#Also add upload file option to input data

with st.sidebar:
    data1_dipdir = st.number_input('Insert DipDir',min_value=0, max_value=359, value=30)
    data1_dip = st.number_input('Insert Dip',min_value=0, max_value=90, value=45)
    data1 = generate_structures(data1_dipdir, 7, data1_dip,5, litho=['Quartzite', 'Itabirite'],size=150)
    data1['structure'] = 'Sb'
    df = data1.copy()

    if st.checkbox('Add 2nd Structure'):

        data2_dipdir = st.number_input('Insert DipDir',min_value=0, max_value=359)
        data2_dip = st.number_input('Insert Dip',min_value=0, max_value=90)
        data2 = generate_structures(data2_dipdir, 7, data2_dip,5, litho=['Schist'],size=50)
        data2['structure'] = 'Sn'
        df = pd.concat([df,data2],ignore_index=True)

    if st.checkbox('Add 3rd Structure'):
        data3_dipdir = st.number_input('Insert DipDir',min_value=0, max_value=359)
        data3_dip = st.number_input('Insert Dip',min_value=0, max_value=90)
        data3 = generate_structures(data3_dipdir, 7, data3_dip,5, litho=['Quartzite', 'Itabirite'],size=50)
        data3['structure'] = 'Fr1'
        df = pd.concat([df,data3],ignore_index=True)




friction =  {'Itabirite': 37, 'Slate':28, 'Schist':30,'Quartzite':35}
df['friction'] = df['litho'].map(friction)

#And must define a few other parameters that characterize the structures

persistency =  {'Sb': 20, 'Sn': 20, 'Fr1':8 ,'Fr2':3}
df['persistency'] = df['structure'].map(persistency)

persistency =  {'Sb': 20, 'Sn': 20, 'Fr1':8 ,'Fr2':3}
df['persistency'] = df['structure'].map(persistency)

spacing =  {'Sb': 1, 'Sn': 0.1, 'Fr1':5 ,'Fr2':8}
df['spacing'] = df['structure'].map(spacing)
# df 


slope_dipdir = st.number_input('Select a Slope Dir',min_value=0, max_value=359, value=30)

slope_dip = st.number_input('Select a Slope Dip', min_value=0, max_value=90, value=45)
st.write(f'The current Slope Atitute is  {slope_dipdir}/{slope_dip}')


slope_height = st.number_input('Select a Slope Height', min_value=0, max_value=None, value=10)
st.write(f'The current Slope height is  {slope_height} meters')

kynematic_window = st.number_input('Select a Kynematic Window', min_value=0, max_value=45, value=20)
st.write(f'The current kynematic window is  {kynematic_window} degrees')




df = planar_analyse(df,slope_dipdir,slope_dip,kynematic_window)
df['apparent_dip'] = df.apply(lambda x: apparent_dip(slope_dipdir, x['dipdir'], x['dip']), axis=1)
df['persistency_height'] = np.sin(np.radians(df['apparent_dip'])) * df['persistency']
df

col1, col2 = st.columns(2)


mss = multi_structure_stereogram(df,web=True)
col1.pyplot(mss,use_container_width=True)

ps = planar_stereogram(df,slope_dipdir, slope_dip, web=True )
col2.pyplot(ps,use_container_width=True)



risk = df.query('planar_rupture == True')
if risk.shape[0] == 0:
    st.write('No Planar Rupture Observable from the inputed Structures and Slope')
else:
    risk = estimate_berm_planar_rupture(risk,slope_dipdir, slope_dip,slope_height)

    samples = risk['projects_minimum_berm']
    ch = cumulative_hist(samples,web=True)
    st.pyplot(ch,use_container_width=True)
