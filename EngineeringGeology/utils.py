import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import pandas as pd
import mplstereonet as mpl
from scipy import stats


def apparent_dip(slope_dipdir,structure_dipdir, structure_dip):
    dip = np.radians(90 - structure_dip)
    a = dipdir_to_xy(structure_dipdir)
    b = dipdir_to_xy(slope_dipdir + 90)
    angle_between_planes = (np.arccos(np.dot(a,b)))
    if angle_between_planes > (np.pi/2):
        angle_between_planes = np.pi - angle_between_planes
    apparent_angle  = np.arctan(np.sin((angle_between_planes))/(np.tan(dip)))
    apparent_angle = np.round(np.degrees(apparent_angle), decimals=1)

    return apparent_angle 
    
def dipdir_to_xy(dipdir):
  dipdir_rad = np.radians(dipdir)
  x = np.cos(dipdir_rad)
  y = np.sin(dipdir_rad)
  norma_decimal = np.sqrt((np.power(x,2) + np.power(y,2)))
  norma = np.around(norma_decimal,decimals=0)
  if norma.any() != 1:
    raise ValueError("Normalized vector other than 1. Expected value:1. Value obtained: {}".format(norma))
  return x,y

def xyz_to_coord(x,y,z):
  norma = np.sqrt((np.power(x,2) + np.power(y,2) + np.power(z,2)))
  x_normalizado = x/norma
  y_normalizado = y/norma
  z_normalizado = z/norma
  norma_normalizada = np.round(np.sqrt((np.power(x_normalizado,2) + np.power(y_normalizado,2) + np.power(z_normalizado,2))))
  if norma_normalizada.any() != 1:
    raise ValueError("Normalized vector other than 1. Expected value:1. Value obtained: {}".format(norma))

  dip = asen(z_normalizado)
  dipdir = np.round(asen(y_normalizado/cos(dip)),2)
  dip = np.round(dip,2)
  

  if x<0  and y >0:
    dipdir = 90 + (90-dipdir)
  else:
    if x<0 and y<0:
      dipdir = dipdir*(-1) + 180
    else:
      if x>0 and y <0:
        dipdir = 270+(dipdir+90)
  
  return dipdir,dip


def coord_to_xyz(dipdir, dip):
  dipdir_rad = np.radians(dipdir)
  dip_rad = np.radians(dip)
  x = np.cos(dipdir_rad) * np.cos(dip_rad)
  y = np.sin(dipdir_rad) * np.cos(dip_rad)
  z = np.sin(dip_rad)
  norma_decimal = np.sqrt((np.power(x,2) + np.power(y,2) + np.power(z,2)))
  norma = np.around(norma_decimal,decimals=0)
  if norma.any() != 1:
    raise ValueError("Normalized vector other than 1. Expected value:1. Value obtained: {}".format(norma))
  return x,y,z


def sen(angulo_graus):
  sen = np.sin(np.radians(angulo_graus))
  return sen

def cos(angulo_graus):
  cos = np.cos(np.radians(angulo_graus))
  return cos

def tan(angulo_graus):
  tan = np.tan(np.radians(angulo_graus))
  return tan

def atan(valor):
  atan = np.degrees(np.arctan(valor))
  return atan

def asen(valor):
  asen = np.degrees(np.arcsin(valor))
  return asen
def acos(valor):
  acos = np.degrees(np.araccos(valor))
  return acos


def generate_structures(mu_dir=60, sigma_dir=7, mu_dip=45,sigma_dip=5, size=500, litho=None):
    behaved =  int(size-size*0.2)
    dr = np.random.normal(mu_dir, sigma_dir,behaved)
    dr = np.array([int(i) for i in dr])
    dp = np.random.normal(mu_dip, sigma_dip, behaved)
    dp = np.array([int(i) for i in dp])

    #add some noise to it
    dipdir = np.hstack([dr, np.random.randint(dr.min()-2*sigma_dir,dr.max()+2*sigma_dir,size-behaved)])
    dip = np.hstack([dp, np.random.randint(dp.min()-2*sigma_dip,dp.max()+2*sigma_dip,size-behaved)])

    if litho is None:
        litho = ['Quartzite','Schist','Itabirite','Slate']

    assert dipdir.shape[0] % len(litho) ==0, 'Size must be proportional to lito len'
    litho = int(dipdir.shape[0]/len(litho)) * litho

    df = pd.DataFrame(data=np.array([dipdir, dip]).T, columns=['dipdir','dip'])
    df['litho']=litho

    for idx in range(df.shape[0]):
        if df.loc[idx,'dip'] > 90 and df.loc[idx,'dipdir'] >= 180:
            df.at[idx,'dipdir'] = df.at[idx,'dipdir'] - 180
            df.at[idx,'dip'] = 180 - df.at[idx,'dip'] 
        elif df.loc[idx,'dip'] > 90 and df.loc[idx,'dipdir'] < 180:
            df.at[idx,'dipdir'] = df.at[idx,'dipdir'] + 180
            df.at[idx,'dip'] = 180 - df.at[idx,'dip']
            
    df.dipdir = np.array([i-360 if i >360 else i for i in  df.dipdir])
    return df 

def generate_full_data():
    mu_dir, sigma_dir = 240,7
    mu_dip,sigma_dip = 40,5
    data1 = generate_structures(mu_dir, sigma_dir, mu_dip,sigma_dip, litho=['Quartzite', 'Itabirite'],size=150)
    data1['structure'] = 'Sb'

    mu_dir, sigma_dir = 220,7
    mu_dip,sigma_dip = 60,5
    datax = generate_structures(mu_dir, sigma_dir, mu_dip,sigma_dip, litho=['Schist'],size=50)
    datax['structure'] = 'Sn'


    mu_dir, sigma_dir = 180,5
    mu_dip,sigma_dip = 55,5
    data2 = generate_structures(mu_dir, sigma_dir, mu_dip,sigma_dip, litho=['Quartzite', 'Itabirite'],size=50)
    data2['structure'] = 'Fr1'

    mu_dir, sigma_dir = 300,10
    mu_dip,sigma_dip = 80,2
    data3 = generate_structures(mu_dir, sigma_dir, mu_dip,sigma_dip, litho=['Quartzite', 'Itabirite'],size=50)
    data3['structure'] = 'Fr2'

    df = pd.concat([data1,datax,data2,data3],ignore_index=True)

    friction =  {'Itabirite': 37, 'Slate':28, 'Schist':30,'Quartzite':35}
    df['friction'] = df['litho'].map(friction)

    #And must define a few other parameters that characterize the structures

    persistency =  {'Sb': 20, 'Sn': 20, 'Fr1':8 ,'Fr2':3}
    df['persistency'] = df['structure'].map(persistency)

    persistency =  {'Sb': 20, 'Sn': 20, 'Fr1':8 ,'Fr2':3}
    df['persistency'] = df['structure'].map(persistency)

    spacing =  {'Sb': 1, 'Sn': 0.1, 'Fr1':5 ,'Fr2':8}
    df['spacing'] = df['structure'].map(spacing)
    return df




def paralellism(dipdir_struct, dipdir_slope, kynematic_window=30):
  a = dipdir_to_xy(dipdir_struct)
  b = dipdir_to_xy(dipdir_slope)
  angle = np.around(np.degrees(np.arccos(np.dot(a,b))),decimals=0)
  if  angle <= kynematic_window:
    return True
  else:
    return False
  
def dip_paralellism(dip_struct, dip_slope, friction_angle):

  if  dip_struct <= dip_slope and dip_struct >= friction_angle:
    return True
  else:
    return False
  
def planar_rupture(paralellism, dip_paralellism):
    if paralellism and dip_paralellism:
        return True
    else:
        return False

def planar_analyse(df,slope_dipdir,slope_dip,kynematic_window): 
    df['paralellism'] = df.apply(lambda x: paralellism(x['dipdir'], slope_dipdir,kynematic_window=kynematic_window), axis=1)
    df['dip_paralellism'] = df.apply(lambda x: dip_paralellism(x['dip'], slope_dip,x['friction']), axis=1)
    df['planar_rupture'] = df.apply(lambda x: planar_rupture(x['paralellism'], x['dip_paralellism']), axis=1)  
    return df                     


def wedge_rupture(wedge_paralellism, dip_paralellism):
    if wedge_paralellism and dip_paralellism:
        return True
    else:
        return False



def get_intersection_dataframe(df, main_structure='Sb'):

    sb = df.query('structure == @main_structure')
    fr = df.query('structure != @main_structure')

    df_intersection = pd.DataFrame(columns=['planes','dipdir1',	'dip1',	'dipdir2',	'dip2',	'weak_friction'])
    
    dipdir1,dip1,dipdir2,dip2,weak_friction,planes = [],[],[],[],[],[]
    for i, (d1,p1,f1,s1) in enumerate(zip(sb.dipdir,sb.dip, sb.friction, sb.structure)):
        
        for x, (d2,p2,f2,s2) in enumerate(zip(fr.dipdir,fr.dip, fr.friction, fr.structure)):
            if f1<=f2:
                weak_friction.append(f1)
            elif f1>f2:
                weak_friction.append(f2)

            dipdir1.append(d1)
            dip1.append(p1)
            dipdir2.append(d2)
            dip2.append(p2)
            planes.append(f'{s1}/{s2}')

    df_intersection['dipdir1'] = dipdir1
    df_intersection['dip1'] = dip1
    df_intersection['dipdir2'] = dipdir2
    df_intersection['dip2'] = dip2
    df_intersection['weak_friction'] = weak_friction
    df_intersection['planes']= planes

    return df_intersection 




def calculate_intersection(df_intersection):
  # Calculate intersections
  bearing_intersect=[]
  plunge_intersect=[]
  for i, (d1,d2,p1,p2) in enumerate(zip(df_intersection.dipdir1,df_intersection.dipdir2,df_intersection.dip1,df_intersection.dip2, )):
    strike_1 = d1 - 90
    strike_2 = d2 - 90
    plunge, bearing = mpl.plane_intersection(strike_1, p1,strike_2, p2)
    plunge = np.round(plunge,2)
    bearing = np.round(bearing,2)
    bearing_intersect.append(bearing)
    plunge_intersect.append(plunge)
  # #Concatena o array de arrays em um só array
  bearing_intersect = np.concatenate(bearing_intersect)
  plunge_intersect = np.concatenate(plunge_intersect)

  return bearing_intersect, plunge_intersect

def estimate_containment_width(area_between_slope_and_rupture_projection,blistered_area,distance_between_slopefoot_and_rupture_projection,distance_between_projection_rupture_and_broken_material ,angle_repose,slope_dip):
  if area_between_slope_and_rupture_projection < blistered_area:
    Lr = (distance_between_slopefoot_and_rupture_projection +
          distance_between_projection_rupture_and_broken_material)
  else:
    Lr = np.sqrt(np.absolute((2*blistered_area*sen(slope_dip-angle_repose))/
          (sen(angle_repose)*sen(180-slope_dip))))
  return Lr

def calculate_remanescent_width_superior_berm(backbreak,berm_width,persistency_height, dip,slope_dip ):
  if backbreak < berm_width:
    Ls = berm_width-(( persistency_height /tan(dip))-
                          (persistency_height/tan(slope_dip)))
  else:
    Ls = "Backbreak > Lo "
  return Ls

def project_minimum_berm(backbreak,required_containment_width,MRFW,MOW):
    if backbreak > required_containment_width:
            x = backbreak + required_containment_width + MRFW
    else:
        if required_containment_width - backbreak >= required_containment_width:
            x = required_containment_width + MRFW
        else:
            x = (required_containment_width + required_containment_width + MRFW -
            (required_containment_width - backbreak))
        
    if x < MOW:
        pmb = MOW
    else:
        if backbreak > required_containment_width:
            pmb = backbreak + required_containment_width + MRFW
        else:
            if required_containment_width - backbreak >= required_containment_width:
                pmb = required_containment_width + MRFW
            else:
                pmb = (required_containment_width + required_containment_width + MRFW -
                        (required_containment_width - backbreak))
    return pmb


def estimate_berm_planar_rupture(risk,slope_dipdir, slope_dip,slope_height):
  PCM = 0  # Pre-collapsed material
  NB = 1  # Number of benches
  AR = 35  # Angle of repose
  BF = 1.4  # Bulking factor
  ODSS = 0  # Operating distance between simple spans
  MRFW = 0  # Minimum required free width
  MOW = 0 # Minimum operational width
  berm_width =  4.5 + (0.2 * slope_height * NB)



  risk['remanecent_height'] = risk.apply(lambda x: slope_height * NB - x['persistency_height'] , axis=1)
  risk['projection_on_slope'] = risk.apply(lambda x: x['remanecent_height'] /  np.sin(np.radians(slope_dip) ), axis=1)

  risk['backbreak'] = (risk['persistency_height']/(np.tan(np.radians(risk['dip']))))-(risk['persistency_height']/(np.tan(np.radians(slope_dip))))
  risk['cutting_area'] = ((np.power(risk['persistency_height'],2)/2)*
          ((1/np.tan(np.radians(risk['dip'])))-
          (1/np.tan(np.radians(slope_dip))))* (1-PCM))

  risk['blistered_area'] = risk['cutting_area'] * BF

  risk['area_between_slope_and_rupture_projection'] = 0.5*np.power(risk['remanecent_height'],2)*(np.sin(np.radians(180-slope_dip))*
                                                  np.sin(np.radians(slope_dip-
                                                                    risk['dip']))/
                                                  np.sin(np.radians(risk['dip'])))
  
  risk['distance_between_slopefoot_and_rupture_projection'] = ((2*risk['area_between_slope_and_rupture_projection'])/
                    (risk['projection_on_slope']*np.sin(np.radians(180-slope_dip))))

  risk['distance_between_projection_rupture_and_broken_material'] =  np.sqrt(np.absolute((2*(risk['blistered_area']-risk['area_between_slope_and_rupture_projection'])*
                  np.sin(np.radians(risk['dip']-AR))/
                    (np.sin(np.radians(180-risk['dip']))*
                    np.sin(np.radians(AR))))))
  
      
  risk['required_containment_width'] = risk.apply(lambda x: estimate_containment_width(x['area_between_slope_and_rupture_projection'],x['blistered_area'],x['distance_between_slopefoot_and_rupture_projection'],x['distance_between_projection_rupture_and_broken_material'] , AR, slope_dip), axis=1 )


  risk['remanescent_width_superior_berm'] = risk.apply(lambda x: calculate_remanescent_width_superior_berm(x['backbreak'],berm_width,x['persistency_height'], x['dip'],slope_dip) , axis=1)

  risk['effective_slope_angle'] = tan(slope_height/
                  (slope_height/tan(slope_dip) +
                  risk['backbreak']+ODSS))


  risk['projects_minimum_berm'] = risk.apply(lambda x: project_minimum_berm(x['backbreak'],x['required_containment_width'], MRFW,MOW) , axis=1)


  return risk