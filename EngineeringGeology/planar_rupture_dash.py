import numpy as np
import mplstereonet as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import *
from plots import *
from scipy import stats
import streamlit as st


st.set_page_config(layout='wide')
np.random.seed(0)
df =generate_full_data()

# df 
slope_dipdir = st.number_input('Insert a Slope Dir',min_value=0, max_value=360)

slope_dip = st.number_input('Insert a Slope Dip', min_value=0, max_value=90)
st.write(f'The current Slope Atitute is  {slope_dipdir}/{slope_dip}')


slope_height = st.number_input('Insert a Slope Height', min_value=0, max_value=None)
st.write(f'The current kynematic window is  {slope_height}')

kynematic_window = st.number_input('Insert a Kynematic Window', min_value=0, max_value=45)
st.write(f'The current kynematic window is  {kynematic_window}')




df = planar_analyse(df,slope_dipdir,slope_dip,kynematic_window)
risk = estimate_berm_planar_rupture(df,slope_dipdir, slope_dip,slope_height)
samples = risk['projects_minimum_berm']




col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

mss = multi_structure_stereogram(df,web=True)
col1.pyplot(mss,use_container_width=True)

ps = planar_stereogram(df,slope_dipdir, slope_dip, web=True )
col2.pyplot(ps,use_container_width=True)

ch = cumulative_hist(samples,web=True)
col3.pyplot(ch,use_container_width=True)