#Requires 2 files: Material inventory per archetype and emissions intensities for each material specification over time.

#import sqlite3
#import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm

#from matplotlib.colors import ListedColormap
#import seaborn as sns

#################################### Import Material Inventory and Quantities for the Archetypes ##################

# Material inventory information
fileM="/Users/johanaforero/Dropbox/Johana/MFA_ZEN_Master_Johana/Model/4.Dynamic mat. analysis/Archetype_inventories_BaselineL.csv"

data_Mat= pd.read_csv(fileM,encoding='utf-8')  # Dataframe type.
data_Mat = data_Mat.fillna(0)
data_Mat = data_Mat.sort_values('Process') #Organizes by the name material specification

# Emission intensities time series data
fileEm="/Users/johanaforero/Dropbox/Johana/MFA_ZEN_Master_Johana/Model/4.Dynamic mat. analysis/TS_Intensities_Ecoinvent.csv"

data_Em= pd.read_csv(fileEm,encoding='utf-8') # Dataframe type. # There may be problems if number separator is ','.  
data_Em = data_Em.sort_values('Process')

#Colors assigned to the materials. From colorbrewer
colors_Mat= np.array(['#4d4d4d','#ffed6f','#9970ab','#e08214','#fdb863','#de77ae','#f1b6da','#b35806','#d9d9d9','#999999','#b2182b','#d8b365'])

##################################### MATERIAL AND EMISSION ANALYSIS FROM INVENTORIES

# Material content for each archetype

### 1. Material use of archetypes grouped by main material

Mat_main = data_Mat.iloc[:,1:].groupby(['Main material']).sum()
Mat_main = Mat_main.sort_values('SFH2019_C',ascending=False) 
mat_order = Mat_main.index.values

Main_materials = Mat_main.sort_values('Main material').index.values
colors_Mat_DF = pd.DataFrame(colors_Mat,index =Main_materials, columns=['color'])

# 2. Emission intensities of main materials categories. As a reference. 
###It is an average of the intensities for the different processes contain in the material categories.

Em_main =data_Em.iloc[:,:3].groupby(['Main material']).mean()
Em_main = Em_main.sort_values('Main material')
Em_main =Em_main.merge(colors_Mat_DF,left_on='Main material',right_index=True)
Em_main = Em_main.sort_values('2019',ascending=False)

### 3. Emission intensities per archetype. 
## Multiplies amount of material specification with its emission intensity. 

CO2_Time =np.zeros([len(data_Em.columns)-2,len(Main_materials),len(arch)]) # (years, materials, archetype)

# The emission intensity from the process is multiplied by the amount of material of the process for each archetype, according to the intensity for each year. 
for index, item in enumerate(CO2_Time):
    
	temp = data_Mat.set_index("Process").iloc[:,1:].multiply(data_Em.set_index("Process").iloc[:,index+1], axis="index")
	temp.index.name = "Process"
	temp = temp.merge(data_Mat.iloc[:,:2], on='Process')
	temp = temp.groupby(['Main material']).sum()
	temp = temp.sort_values('Main material') # the array is sorted alphabetically by material.
    
	#convert to array
	temp_array = temp.values ## (materials)x(archetypes)
	#add to array (material, ak, i)
	CO2_Time[index][:][:] = temp_array # years, materials, archetype, # array with total emissions per AK in time # materials are sorted alphabetically

# CO2 emissions by archetype for first year of analysis, . 
CO2_tot_ak = array_to_DF( CO2_Time[0,:,:], Main_materials, 1, Mat_main.columns.values, 2)
CO2_tot_ak_C = CO2_tot_ak.merge(colors_Mat_DF,left_index=True,right_index=True)
CO2_tot_ak_sort = CO2_tot_ak_C.sort_values('Total',ascending=False)
co2_order = CO2_tot_ak_sort.index.values

##########################################  PLOTS  
##########################################  Plot material use and emission intensities by archetype: 

# On one single figure:

#Plot of Material use by archetype 

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(221)
Mat_main_pl = Mat_main.merge(colors_Mat_DF,left_index=True,right_index=True)

Mat_main_pl.iloc[:,:-1].T.plot(kind='bar', stacked=True, color = Mat_main_pl['color'].values, ax = ax, width=0.65, alpha=1)

ax.get_legend().remove()
plt.xlabel('Archetype')
plt.ylabel('kg/$\mathregular{m^2}$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


#Plot of emission intensities by archetype, year 2019.
ax = fig.add_subplot(222)
CO2_tot_ak_sort_mat = CO2_tot_ak_sort.T[mat_order]
CO2_tot_ak_sort_mat.iloc[:-2,:].plot(kind='bar', stacked=True,  color = CO2_tot_ak_sort_mat.T['color'].values, width=0.65, ax=ax)

plt.xlabel('Archetype')
plt.ylabel('kg of $\mathregular{CO_2e}$/$\mathregular{m^2}$')
plt.legend(loc='upper left', bbox_to_anchor=(1.02,1))

#Save figure settings
nameFig = 'Mat_em_ak_intensities.'
pathR_PDF='/Users/johanaforero/Dropbox/Johana/MFA_ZEN_Master_Johana/Model/4.Dynamic mat. analysis/Results_figures/'
#plt.savefig(pathR_PDF+nameFig+'pdf',bbox_inches='tight')

########################################## Plot emission intensities by material.2019: 

# Plot emission intensities for the materials. 
title_Em= "Emission intensities of material categories for 2019"
fig, ax = plt.subplots()
Em_main['2019'].plot(kind='bar', title=title_Em,  color=Em_main['color'].values, legend=None, ax = ax)

plt.xlabel('Materials')
plt.ylabel('kg of $\mathregular{CO_2e}$/kg')

##########################################  PLOTS END 

############################## DYNAMIC ANALYSIS OF MATERIALS AND EMISSIONS

### MATERIALS

### Create array of material
# array of quantity of material with the archetypes corresponding to each activity
# Material kg/m2
array_M = Mat_main.sort_values(by ='Main material').values 

array_M_C =array_M[:][:,ak_C-1] 
array_M_R =array_M[:][:,ak_R-1] 

ak_D_1=np.array([7,7,10,10,13,13]) #has to be adjusted every time, accoridng to the archetype with the material content for the demolished archetypes.

array_M_D =array_M[:][:,ak_D_1-1] #
array_M_C_new = array_M[:][:,14] # 


######################### Create 3D arrays of materials and activities ######################
# material use per m2
Mat_C =np.zeros([len(year_C),len(array_M),len(ak_C)]) # Create 3D array (year, archetype, material use)
Mat_R =np.zeros([len(year_R),len(array_M),len(ak_R)])
Mat_D =np.zeros([len(year_D),len(array_M),len(ak_D)])

Mat_C_new =np.zeros([len(year_D),len(array_M)]) # year, mat

for index, item in enumerate(C_m2):
    for ind, it in enumerate(array_M_C):
        Mat_C[index][ind][:] = C_m2[index][:]*array_M_C[ind][:]  #year,material,ak

for index, item in enumerate(R_m2):
    for ind, it in enumerate(array_M_R):
        Mat_R[index][ind][:] = R_m2[index][:]*array_M_R[ind][:]  #year,material,ak

for index, item in enumerate(D_m2):
    for ind, it in enumerate(array_M_D):
        Mat_D[index][ind][:] = D_m2[index][:]*array_M_D[ind][:]  #year,material,ak 

for index, item in enumerate(C_new_m2):
	 Mat_C_new[index][:] = C_new_m2[index]*array_M_C_new[:]
    
re_factor_D = 1 #0.20

# Year X material (kg of material) 

Mat_C_t_m = array_to_DF( Mat_C, year_C, 1, Main_materials, 2)
Mat_R_t_m = array_to_DF( Mat_R, year_R, 1, Main_materials, 2)
Mat_D_t_m = array_to_DF( Mat_D, year_D, 1, Main_materials, 2)

Mat_C_new_t_m = array_to_DF( Mat_D, year_D, 1, Main_materials, 2)

# year x archetype (kg of material) 
Mat_C_t_ak = array_to_DF( Mat_C, year_C, 1, np.array(labels_C), 1)
Mat_R_t_ak = array_to_DF( Mat_R, year_R, 1, np.array(labels_R), 1)
Mat_D_t_ak = array_to_DF( Mat_D, year_D, 1, np.array(labels_D), 1)


###############################################  PLOT Storyline use of material (C,R,D)
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

# if not discrete then delete sortedM_disc

#Construction
sortedM=Mat_C_t_m.iloc[:-2,:-1][mat_order] #organized according to material
colorsM = sortedM.T.merge(colors_Mat_DF,left_index=True,right_index=True)
#sortedM.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax=ax,linewidth =0)
sortedM_disc= years_to_range(sortedM)
sortedM_disc.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax=ax,linewidth =0)

#Renovation + new construction 
ren_cons = Mat_R_t_m.iloc[:-2,:-1].add(Mat_C_new_t_m.iloc[:-2,:-1],fill_value=0)

sortedM = ren_cons[mat_order]
colorsM = sortedM.T.merge(colors_Mat_DF,left_index=True,right_index=True)
#sortedM.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax = ax,linewidth =0)
sortedM_disc= years_to_range(sortedM)
sortedM_disc.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax = ax,linewidth =0)

#Out renovation and demolition:
ren_cons = Mat_R_t_m.iloc[:-2,:-1].add(Mat_C_new_t_m.iloc[:-2,:-1],fill_value=0)
sortedM = ren_cons[mat_order]
colorsM = sortedM.T.merge(colors_Mat_DF,left_index=True,right_index=True)
#sortedM.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax = ax,linewidth =0)
sortedM_disc= years_to_range(sortedM*-1)
sortedM_disc.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax = ax,linewidth =0,alpha=0.5)

plt.xlabel('year')
plt.ylabel('kg')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax=plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(-5500000,15000000)
plt.xlim(2019,2081)

handles, labels = plt.gca().get_legend_handles_labels()
hand_lab =list(zip(labels,handles))
hand_lab = pd.DataFrame (hand_lab, columns=['labels','handles'])
hand_lab = hand_lab.drop_duplicates(subset="labels")
hand_lab = hand_lab.set_index('labels')
plt.legend(labels = hand_lab.index.values.tolist(), handles = hand_lab['handles'].values.tolist(),loc ='upper right', bbox_to_anchor=(1.35,1))

#Saving settings
nameFig = 'Storyline_material_use.'
#plt.savefig(pathR_PDF+nameFig+'pdf',bbox_inches='tight')

############################################# EMISSIONS #################################################

array_CO2_C =CO2_Time[year_C-2019,:,:] 
array_CO2_C = array_CO2_C[:,:,ak_C-1] # year, material, archetype 

array_CO2_R =CO2_Time[year_R-2019,:,:]
array_CO2_R = array_CO2_R[:,:,ak_R-1] # year, material, archetype

array_CO2_D =CO2_Time[year_D-2019,:,:] 
array_CO2_D = array_CO2_D[:,:,ak_D_1-1] # year, material, archetype

######################### Create 3D arrays of materials and activities ######################

CO2_C =np.zeros([len(year_C),len(Main_materials),len(ak_C)]) # Create 3D array (year, material, ak) 
CO2_R =np.zeros([len(year_R),len(Main_materials),len(ak_R)]) # 
CO2_D =np.zeros([len(year_D),len(Main_materials),len(ak_D)]) # 

for index, item in enumerate(C_m2): 
    
    for ind, it in enumerate(array_CO2_C[index,:,:]): 
        CO2_C[index][ind][:] = C_m2[index][:]*array_CO2_C[index][ind][:]  #year,material,ak

for index, item in enumerate(R_m2):
    for ind, it in enumerate(array_CO2_R[index,:,:]):
        CO2_R[index][ind][:] = R_m2[index][:]*array_CO2_R[index][ind][:]  #year,material,ak

for index, item in enumerate(D_m2):
    for ind, it in enumerate(array_CO2_D[index,:,:]):
        CO2_D[index][ind][:] = D_m2[index][:]*array_CO2_D[index][ind][:]   #year,material,ak

#Dataframes to be build, per activity and then together:

re_factor_D = 1 #0.20

# year X material (kg of CO2)
CO2_C_t_m = array_to_DF( CO2_C, year_C, 1, Main_materials, 2)
CO2_R_t_m = array_to_DF( CO2_R, year_R, 1, Main_materials, 2)
CO2_D_t_m = array_to_DF( CO2_D, year_D, re_factor_D , Main_materials, 2)

# year X archetype (kg of CO2)
CO2_C_t_ak = array_to_DF( CO2_C, year_C, 1, np.array(labels_C), 1)
CO2_R_t_ak = array_to_DF( CO2_R, year_R, 1, np.array(labels_R), 1)
CO2_D_t_ak= array_to_DF( CO2_D, year_D, re_factor_D , np.array(labels_D), 1)


###############################################  PLOT Storyline Emissions

#PLOT STORYLINE CO2
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

order =  mat_order # co2_order. Order according to materials use.

sortedM = CO2_C_t_m.iloc[:-2,:-1][order]
colorsM = sortedM.T.merge(colors_Mat_DF,left_index=True,right_index=True)
#sortedM.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax=ax,linewidth =0)
sortedM_disc= years_to_range(sortedM)
sortedM_disc.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax=ax,linewidth =0)

ren_con = CO2_R_t_m.iloc[:-2,:-1].add(CO2_D_t_m.iloc[:-2,:-1],fill_value=0)
sortedM = ren_con[order]
colorsM = sortedM.T.merge(colors_Mat_DF,left_index=True,right_index=True)
#sortedM.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax = ax,linewidth =0)
sortedM_disc= years_to_range(sortedM)
sortedM_disc.plot(kind='area', stacked=stacked, color = colorsM['color'].values, ax = ax,linewidth =0)

plt.xlabel('year')
plt.ylabel('kg of $\mathregular{CO_2e}$')
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax=plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylim(0,6000000)
plt.xlim(2019,2081)

handles, labels = plt.gca().get_legend_handles_labels()
hand_lab =list(zip(labels,handles))
hand_lab = pd.DataFrame (hand_lab, columns=['labels','handles'])
hand_lab = hand_lab.drop_duplicates(subset="labels")
hand_lab = hand_lab.set_index('labels')

plt.legend(labels = hand_lab.index.values.tolist(), handles = hand_lab['handles'].values.tolist(),loc ='upper left', bbox_to_anchor=(1,1))

#Saving settings
nameFig = 'Storyline_emissions.'
#plt.savefig(pathR_PDF+nameFig+'pdf',bbox_inches='tight')

###############################################  PLOT  Cumulative material and emissions 

fig = plt.figure(figsize=(14,8))

ax = fig.add_subplot(221)

Mat_R_nC_t_m =  Mat_R_t_m.iloc[:-2,:-1].add(Mat_C_new_t_m.iloc[:-2,:-1],fill_value=0)
cum_mat = pd.concat([Mat_C_t_m.iloc[:-2,:-1], Mat_R_nC_t_m])
cum_mat = cum_mat.cumsum()

for i in range(2031,2035):
    cum_mat.loc[i, :] = cum_mat.loc[2030]

cum_mat = cum_mat.sort_index()
cum_mat = pd.concat([cum_mat, colors_Mat_DF.T])
cum_mat_sorted = cum_mat.loc[2080].sort_values(ascending = False).index.values
cum_mat = cum_mat[cum_mat_sorted]
cum_mat.iloc[:-1,:].plot(kind='area', stacked=stacked, color = cum_mat.loc['color'].values,linewidth =0, ax=ax)

plt.legend(loc=2, bbox_to_anchor=(1,1))
plt.xlabel('year')
plt.ylabel('Cumulative kg of materials')
plt.xlim(2019,2080)
plt.title('Cumulative amount of materials over time')
ax.get_legend().remove()


ax = fig.add_subplot(222)

CO2_R_D_t_m =  CO2_R_t_m.iloc[:-2,:-1].add(CO2_D_t_m.iloc[:-2,:-1],fill_value=0)
cum_em = pd.concat([CO2_C_t_m.iloc[:-2,:-1], CO2_R_D_t_m])
cum_em = cum_em.cumsum()

for i in range(2031,2035):
    cum_em.loc[i, :] = cum_em.loc[2030]

cum_em = cum_em.sort_index()
cum_em = pd.concat([cum_em, colors_Mat_DF.T])
cum_co2_sorted = cum_em.loc[2080].sort_values(ascending = False).index.values
cum_em = cum_em[cum_mat_sorted]
cum_em.iloc[:-1,:].plot(kind='area', stacked=stacked, color = cum_em.loc['color'].values,linewidth =0, ax = ax)

plt.legend(loc=6, bbox_to_anchor=(-0.9,-0.35),ncol=4)
plt.xlabel('year')
plt.ylabel('Cumulative kg of $\mathregular{CO_2e}$')
plt.xlim(2019,2080)
plt.title('Cumulative embodied emissions over time')

#Saving settings
nameFig = 'Cumulative material and emissions.'
#plt.savefig(pathR_PDF+nameFig+'pdf',bbox_inches='tight')

###############################################  PLOT END 

#### Print total emissions:

Total_construction = CO2_C_t_m['Total']['Total']
print('Total construction emissions = '+ str(Total_construction) )

Total_renovation = CO2_R_t_m['Total']['Total']
print('Total renovation emissions = '+ str(Total_renovation) )

Total_new_construction = CO2_D_t_m['Total']['Total']
print('Total new construction emissions = '+ str(Total_new_construction) )

Total =Total_construction + Total_renovation + Total_new_construction
print('Total  emissions = '+ str(Total) )

Total_per_year = Total/(2080-2019)
print('Total  emissions per year = '+ str(Total_per_year) )
#################### END ##################################








