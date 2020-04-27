# This code requires as input a database with the detailed building stock 'Baseline.db'
# Baseline.db organizes the building stock in 4 tables: Arhcetype, Construction, Renovation, Deconstruction.
#


import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

#Style of graphics
import seaborn as sns
sns.set()
sns.set_style("ticks")
sns.set_context(rc = {'patch.linewidth': 0.0})

################## Import information of buildings from SQLite as pandas ##################

### The location of the file Baseline.db has to be included in path.
path = '.../Baseline.db' #needs to be set

con = sqlite3.connect(path)
cur = con.cursor()

# Return results of query. Buildings are given as units. 

#Archetype description:
df_Ak = pd.read_sql_query('SELECT Archetype_ID, Cohort_ID, Cohort_description, Type_ID, Type_name, State_ID, Description FROM Archetype', con)
print (df_Ak)

#Archetypes
arch =pd.read_sql_query('SELECT Archetype_ID FROM Archetype', con)# all archetypes
ak_C=pd.read_sql_query('SELECT DISTINCT Archetype_ID FROM Construction ORDER BY Archetype_ID',con)
ak_R=pd.read_sql_query('SELECT DISTINCT New_Archetype_ID FROM Renovation ORDER BY New_Archetype_ID ',con)
ak_D=pd.read_sql_query('SELECT DISTINCT Last_Archetype_ID FROM Deconstruction ORDER BY Last_Archetype_ID',con)

# Construction data
data_C=pd.read_sql_query('SELECT Construction_year, Archetype_ID, count(*) FROM Construction GROUP BY Archetype_ID, Construction_year', con)
# Renovation data
data_R=pd.read_sql_query('SELECT Renovation_year, New_Archetype_ID, COUNT(*) FROM Renovation GROUP BY Renovation_year, New_Archetype_ID', con)
# Demolition data
data_D=pd.read_sql_query('SELECT Deconstruction_year, Last_Archetype_ID, count(*) FROM Deconstruction GROUP BY Deconstruction_year, Last_Archetype_ID', con)

# close the connection
con.close()

################## 1. Create arrays for Archetypes, years, C=construction, R=renovation, D=demolition ##################

# Convert pandas to numpy arrays as: array([[year, arch, quantity],[year, arch, quantity],...])
ak_C=np.asarray(ak_C)
ak_R=np.asarray(ak_R)
ak_D=np.asarray(ak_D)

# Flaten the arrays
ak_C =ak_C.flatten();
ak_R =ak_R.flatten();
ak_D =ak_D.flatten();

# Vector of years for each activity:
year_C = np.arange(data_C['Construction_year'].min(),data_C['Construction_year'].max()+1)
year_R = np.arange(data_R['Renovation_year'].min(),data_R['Renovation_year'].max()+1)
year_D = np.arange(data_D['Deconstruction_year'].min(),data_D['Deconstruction_year'].max()+1)

# Activities to numpy array
data_C =np.asarray(data_C)
data_R =np.asarray(data_R)
data_D =np.asarray(data_D)

# Dimension: number of years and number of archetypes.
dim_C=[len(year_C),len(arch)]
dim_R=[len(year_R),len(arch)]
dim_D=[len(year_D),len(arch)]

# Initialize vectors
C = np.zeros(dim_C) # Construction
R = np.zeros(dim_R) # Renovation
D = np.zeros(dim_D) # Demolition

# Floor useful area  per archetype. FA. 

unitSch = 6474 # School
unitKng = 2140# Kndg
unitSFH = 80 # SFH

# Array with floor area per archetype. 
FA = [unitKng, unitKng,unitKng,  unitSch,  unitSch,unitSch,  unitSFH, unitSFH, unitSFH,  unitSFH,  unitSFH,  unitSFH,  unitSFH, unitSFH,  unitSFH]
FA = np.array(FA)

# Array with labels
ak ='AK'
#15 ak
labels_AK = ['Kind_C','Kind_R1', 'Kind_R2', 
             'School_C','School_R1','School_R2',
             'SFH2019_C','SFH2019_R1','SFH2019_R2',
             'SFH2021_C','SFH2021_R1','SFH2021_R2',
             'SFH2026_C','SFH2026_R1','SFH2026_R2']
labels_C = ['Kind_C','School_C','SFH2019_C','SFH2021_C','SFH2026_C']
labels_R = ['Kind_R1', 'Kind_R2',
            'School_R1','School_R2',
            'SFH2019_R1','SFH2019_R2',
            'SFH2021_R1','SFH2021_R2',
            'SFH2026_R1','SFH2026_R2']
labels_D = ['SFH2019_R1','SFH2019_R2',
            'SFH2021_R1','SFH2021_R2',
            'SFH2026_R1','SFH2026_R2']

################## Fill in quantities in 2D arrays for each year and archetype ##################

#Numpy arrays
def fill_array (X, data_X, year_X):
	for i in data_X:
		j = i[1]-1 # data[i,1] refers to the archetype
		y =np.where (year_X == i[0]) # data[i,0] refers to the year
		X[y,j] = i[2]
	return X

C = fill_array (C, data_C, year_C)
C= C[:][:,ak_C-1] #Actual dimension according to archetypes
C_m2 = C*FA[:][ak_C-1]#Data in m2

R = fill_array (R, data_R, year_R)
R= R[:][:,ak_R-1] #Actual dimension according to archetypes
R_m2 = R*FA[:][ak_R-1]# Data in m2

D = fill_array (D, data_D, year_D)
D= D[:][:,ak_D-1] #Actual dimension according to archetypes
D_m2= D*FA[:][ak_D-1]# Data in m2 (m2 demolished.)

#To dataframe:

def array_to_DF( X, year_X, factor, colName, axisSum):
	n=len(X.shape)
	if n==3:
		#grouping, summing 
		df = pd.DataFrame(np.sum(X,axis=axisSum)*factor,index=year_X,columns=colName)
		df = df.groupby(df.columns.values, axis=1).sum()
	else: 
		df = pd.DataFrame(X, index=year_X, columns=colName)

	y = df.iloc[:,0:].mean(axis=0).to_frame().transpose().rename(index={0:'Average'})
	df = pd.concat([df,y])
	x=df.iloc[:-1,0:].sum(axis=0).to_frame().transpose().rename(index={0:'Total'})
	df = pd.concat([df,x])
	df['Total']=df.sum(axis=1)
	return df

#Final dataframes with area by year and archetype. Used in the second part, dynamicMaterialAnalysis.py
C_m2_df = array_to_DF( C_m2, year_C, 1, np.array(labels_C), 1)
R_m2_df = array_to_DF( R_m2, year_R, 1, np.array(labels_R), 1)
D_m2_df = array_to_DF( D_m2, year_D, 1, np.array(labels_D), 1) 


############################################  Storyline PLOTS  ###################################################


stacked = 'True'

#The number of colors need to be adjusted if the number of archetypes changes. Move and delete colors if a 
arch_Palette = ['#756bb1','#bcbddc','#efedf5',
                '#e6550d','#fdae6b','#fee6ce',
                '#3182bd','#9ecae1','#deebf7',
                '#31a354','#a1d99b','#e5f5e0',
                '#c51b8a','#fa9fb5','#fde0dd'] # Array of colors for 15 archetypes.

# Assigned colors to archetypes and by activities.
colorsAK = pd.DataFrame(arch_Palette,index =labels_AK, columns=['color'])

colors = np.array(arch_Palette)
colors_C = colors[ak_C-1]
colors_R = colors[ak_R-1]
colors_D = colors[ak_D-1]

#Construction, Renovation and Demolition together. 

def years_to_range(X_m2_df_iloc):
    
   #discrete range of years 
    a = X_m2_df_iloc
    a.index.name = 'year'
    a = a.reset_index()
    a1 = a.copy()
    a1['year']=a1['year']+1
    b = a.append(a1)
    b = b.rename_axis('index').sort_values(['index', 'year'], ascending=[True, True])
    b = b.set_index('year')
    return b

### PLOT Storyline by parts

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)

#construction
C_Splot = years_to_range(C_m2_df.iloc[:-2,:-1])
x = pd.to_numeric(C_Splot.index.values)
y = C_Splot.values.T
ax.stackplot(x,y, labels=labels_C, colors=colors_C, alpha =alpha)

#New constructed due to demolition, to keep the area balance
C_new_m2_df = D_m2_df['Total'].iloc[:-2].to_frame(name='SFH_new_C') 

#Renovation and new construction
R_plus_new = R_m2_df.iloc[:-2,:-1].join(C_new_m2_df).fillna(0)
R_Splot = years_to_range(R_plus_new)
x = pd.to_numeric(R_Splot.index.values)
y = R_Splot.values.T

C_new_m2 = C_new_m2_df.values

labels_R_N = labels_R.copy()
labels_R_N.append('SFH_new_C')
colors_R_N = np.append(colors_R,'#e7298a')

ax.stackplot(x,y, labels=labels_R_N, colors=colors_R_N, alpha =alpha)

#Demolition
D_m2_df_neg = D_m2_df.iloc[:-2,:-1]*-1
D_Splot = years_to_range(D_m2_df_neg)
x = pd.to_numeric(D_Splot.index.values)
y = D_Splot.values.T
ax.stackplot(x,y, labels=labels_D, colors=colors_D, alpha =alpha)

plt.xlabel('year')
plt.ylabel('$\mathregular{m^2}$')
plt.xlim(2019,2081)
plt.ylim(-6000, 18000)

handles, labels = plt.gca().get_legend_handles_labels()
hand_lab =list(zip(labels,handles))
hand_lab = pd.DataFrame  (hand_lab, columns=['labels','handles'])
hand_lab = hand_lab.drop_duplicates(subset="labels")
hand_lab = hand_lab.set_index('labels')
labelsAK = np.array(labels_AK)
labelsAK= np.append(labelsAK,'SFH_new_C')
hand_lab = hand_lab.T[labelsAK].T

plt.legend(labels = hand_lab.index.values.tolist(), handles = hand_lab['handles'].values.tolist(),loc =6,  bbox_to_anchor=(-0.25,-0.3), ncol=6)

#Save figure settings
nameFig = 'Storyline dynamic stock.'
pathR_PDF='.../Results_figures/' # Need to be set
plt.savefig(pathR_PDF_S+nameFig+'pdf',bbox_inches='tight')
