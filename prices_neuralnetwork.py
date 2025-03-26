import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import R2Score
#!
#!
pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', None)
np.set_printoptions(suppress=True, precision=6)
%autoindent OFF  # just a NeoVIM thing, you can comment this out.

# As always, start importing training and test datasets, also joining them for data wrangling
# IMPORTANT! Be careful in dealing with both sets joined, there may be leakage of information
# to one another
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#!
# Dimensions adjustment - adding SalePrice feature in test df
test['SalePrice'] = 0
#!
# Adding origin informations
test['Origem'] = 'Teste'
train['Origem'] = 'Treino'
#!
# Joining both datasets. WARNING! Be careful dealing with this, to avoid leakage.
df_complete = pd.concat([train, test])
#!
#########################################################################################3
#!
#!
#####
# CATEGORICAL FEATURES
#####
#!
# Segregating categorical and numeric features, for both train and test datasets
# They must be treated independently, to avoid data leakage from one another, whenever makes sense
#!
colunas_quali = df_complete.columns[df_complete.dtypes == 'object'] # filters the categorical features
variaveis_quanti = df_complete.drop(colunas_quali, axis = 1) # selects only the numeric features
#!
colunas_quanti = df_complete.columns[df_complete.dtypes != 'object'] # filters the numeric features
variaveis_quali = df_complete.drop(colunas_quanti, axis = 1) # selects only the categorical features
#!
# Reinserting origin feature and reseting the indexes
variaveis_quanti['Origem'] = variaveis_quali['Origem']
variaveis_quanti.reset_index(inplace=True, drop=True)
variaveis_quali.reset_index(inplace=True, drop=True)
variaveis_quali['Id'] = variaveis_quanti['Id']  # it's always a good idea to keep a second reference (index)
#variaveis_quanti = variaveis_quanti.drop(['index', 'Id'], axis = 1)
#variaveis_quali = variaveis_quali.drop('index', axis = 1)
#!
# Some features have NaN values, that'll be treated properly. The decision process on each imputation is handled on a case-by-case basis.
nulos = variaveis_quali.isnull().sum()
nulos_ordenados = (nulos.sort_values(ascending=False)/len(variaveis_quali))*100
print(nulos_ordenados.head(10)) # the top 5 features have over 20% NaN values, we'll treat them below
#!
### Description of the features with the most NaNs:
# - PoolQC: NaN values indicate the absence of a pool (then input 'None')
# - MiscFeature: miscellaneous, NaN indicates the absence of this type of feature (input 'None')
# - Alley: type of alley/street for access to the house. If the house has an exit to the main street, the feature cannot even exist (input 'None')
# - Fence: quality of the house's fence, if the house does not have a fence, this information is obviously NaN (input 'None')
# - FireplaceQu: indicates the quality of the fireplace(s) in the house. If there's no fireplaces, this feature cannot exist (input 'None')
imputar_none = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
variaveis_quali[imputar_none] = variaveis_quali[imputar_none].fillna('None')
#!
#!
### Let's check the other features
nulos = variaveis_quali.isnull().sum() # let's get the remaining features that have NaN values
nulos_ordenados = nulos.sort_values(ascending=False)
temp = pd.DataFrame(nulos_ordenados)
mask = temp[0].isin([0])
print(temp[~mask].index) # prints said features
#!
## Features that we'll impute 'None' straight away:
# - 'Garage' features: classify the state of conservation, finish and location of the garage, if any. If there is no garage, the feature has a NaN value.
# - 'Bsmt' features: classify the level of quality, exposure and finish of the basement, if any. NaN indicates that there is no basement in the house.
# - Exterior1st: exterior covering on house. We'll assume that NaN indicates nonexistence, 'None' imputed.
# - Exterior2nd: exterior covering on house (if more than one material) (same reasoning of the above).
# - MasVnrType: Masonry veneer type. We'll assume that nonexistent information corresponds to 'None'.
imputar_none2 = ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1','Exterior1st', 'Exterior2nd', 'MasVnrType']
variaveis_quali[imputar_none2] = variaveis_quali[imputar_none2].fillna('None')
#!
# Let's re-check the garages, to verify that all 'None' quality in fact has all 'None' type
variaveis_quali[(variaveis_quali['GarageQual'] == 'None') & (variaveis_quali['GarageType'] != 'None')]  # Two observations has wrong records, let's correct that
garages_notnone = variaveis_quali[(variaveis_quali['GarageQual'] == 'None') & (variaveis_quali['GarageType'] != 'None')].index.values  # fetch the indexes
variaveis_quali.loc[garages_notnone, 'GarageType'] = 'None'  # and register the right info, easy peasy
#!
#!
## Deliberated imputations for the classification features:
# - Functional: Home functionality (assume typical unless deductions are warranted). This info is from database origin. NaN imputed 'Typ'.
# - SaleType: Type of sale, imputed the mode based on neighborhood.
# - Utilities: Types of basic services available (water, electricity, gas, etc.). Imputed as the neighborhood mode.
# - KitchenQual: Kitchen quality. Imputed as neighborhood mode.
# - Electrical: Electrical system. Imputed as neighborhood mode.
# - MSZoning: Identifies the general zoning classification of the sale. Since this is zoning, it is assumed that the zoning in the same neighborhood is the same (no NaN in the description).
#!
# Since we're gonna start dealing with information that may leak information from train/test
# sets to one another, we need treat them independently. I'll illustrate my checking process
#!
# Functional has metainformation from original database, easy:
variaveis_quali['Functional'] = variaveis_quali['Functional'].fillna('Typ')
#!
# In Utilities feature we'll use neighborhood modes, so we'll segregate train/test:
variaveis_quali['Utilities'].isnull().sum()  # check for NaNs: yes, there are two.
variaveis_quali[variaveis_quali['Origem'] == 'Treino']['Utilities'].isnull().sum()  # check in train dataset, nope.
variaveis_quali[variaveis_quali['Origem'] == 'Teste']['Utilities'].isnull().sum()  # check in test dataset, here they are.
#!
onde = np.where(variaveis_quali[variaveis_quali['Origem'] == 'Teste']['Utilities'].isnull())[0]
variaveis_quali.loc[onde, 'Neighborhood']  # so, first obs (455) is in NWAmes and the second (485) is in NAmes neighborhood
mode_1 = variaveis_quali[(variaveis_quali['Origem'] == 'Teste') & (variaveis_quali['Neighborhood'] == 'NWAmes')]['Utilities'].mode()  # returns AllPub
mode_2 = variaveis_quali[(variaveis_quali['Origem'] == 'Teste') & (variaveis_quali['Neighborhood'] == 'NAmes')]['Utilities'].mode()  # also returns AllPub
#!
# Then the imputation is simple:
variaveis_quali['Utilities'] = variaveis_quali['Utilities'].fillna(mode_1[0])  # we can use any mode objects, since they're the same thing here
#!
#!
# SaleType procedures
variaveis_quali[variaveis_quali['Origem'] == 'Teste']['SaleType'].isnull().sum()
onde = np.where(variaveis_quali[variaveis_quali['Origem'] == 'Teste']['SaleType'].isnull())[0]
bairro = variaveis_quali.loc[onde, 'Neighborhood'].values
mode_1 = variaveis_quali[(variaveis_quali['Origem'] == 'Teste') & (variaveis_quali['Neighborhood'] == bairro[0])]['SaleType'].mode()  # returns WD
variaveis_quali['SaleType'] = variaveis_quali['SaleType'].fillna(mode_1[0])
#!
#MSZoning
variaveis_quali[variaveis_quali['Origem'] == 'Teste']['MSZoning'].isnull().sum()  # four obs, all in test df
onde = np.where(variaveis_quali[variaveis_quali['Origem'] == 'Teste']['MSZoning'].isnull())[0]  # 455, 756, 790, 14444
bairros = variaveis_quali.loc[onde, 'Neighborhood'].values  # NWAmes, CollgCr, Blmngtn, CollgCr
mode_1 = variaveis_quali[(variaveis_quali['Origem'] == 'Teste') & (variaveis_quali['Neighborhood'] == bairros[0])]['MSZoning'].mode()  # all obs in 'bairros' return RL
variaveis_quali['MSZoning'] = variaveis_quali['MSZoning'].fillna(mode_1[0])
#!
# This step regards imputation for 'KitchenQual':
variaveis_quali[variaveis_quali['Origem'] == 'Teste']['KitchenQual'].isnull().sum()  # just one obs, also in test df
onde = np.where(variaveis_quali[variaveis_quali['Origem'] == 'Teste']['KitchenQual'].isnull())[0]
bairro = variaveis_quali.loc[onde, 'Neighborhood'].values
mode_1 = variaveis_quali[(variaveis_quali['Origem'] == 'Teste') & (variaveis_quali['Neighborhood'] == bairro[0])]['KitchenQual'].mode()  # returns Gd
variaveis_quali['KitchenQual'] = variaveis_quali['KitchenQual'].fillna(mode_1[0])
#!
# Electrical feature
variaveis_quali[variaveis_quali['Origem'] == 'Treino']['Electrical'].isnull().sum()  # just one obs, this time in train df
onde = np.where(variaveis_quali[variaveis_quali['Origem'] == 'Treino']['Electrical'].isnull())[0]
bairro = variaveis_quali.loc[onde, 'Neighborhood'].values
mode_1 = variaveis_quali[(variaveis_quali['Origem'] == 'Treino') & (variaveis_quali['Neighborhood'] == bairro[0])]['Electrical'].mode()  # returns SBrkr
variaveis_quali['Electrical'] = variaveis_quali['Electrical'].fillna(mode_1[0])
#!
# Data wrangling on categorical features is done here. Now that we have our object 'variaveis_quali' (the categoricals), let's move on to the numerical features
# If you wanna check for any remaining NaNs just uncomment the line below and run it
#variaveis_quali.isnull().sum()
#!
#!
#########################################################################################3
#!
#!
#####
# NUMERICAL FEATURES
#####
#!
# Again, some features (11) have NaN values, that we'll deal with them now
nulos_quanti = variaveis_quanti.isnull().sum()
nulos_ordenados_quanti = (nulos_quanti.sort_values(ascending=False)/len(variaveis_quanti))*100
print(nulos_ordenados_quanti.head(12))
#!
# Adjusted variables:
# - GarageArea - size of the garage, using the median of garages with the same GarageType, if there is a garage in the house.
# - GarageYrBlt - Year the garage was built. If there is a garage built after the last renovation (GarageYrBlt > YearRemodAdd), it is established that the greater year is the year of the last remodeling. The GarageYrBlt variable is used to account for the age of the house (the smaller, the better; it has a positive impact on the price).
# - LotFrontage - Dimension of the front of the lot, in feet. Calculated by dividing LotArea by the average of the LotArea/LotFrontage division of the pre-existing values.
# - MasVnrArea - Masonry veneer area in square feet. Direct relationship with MasVnrType (qualitative variable) - if this is non-existent (as all NaN values here), then the area is 0.
# - GarageCars - size of the garage in terms of capacity of quantity of cars, using the average of other garages depending on the garage area.
# - Bsmt*** - NaN values exist when there are no basements, zero is registered.
#!
# Dealing with 'GarageArea' feature
variaveis_quanti['GarageArea'].isnull().sum() # there's just one NaN
pos_garage = variaveis_quanti['GarageArea'][variaveis_quanti['GarageArea'].isnull()].index[0] # just one observation, 2576
variaveis_quanti.loc[pos_garage]  # here you can verify the garage has no area, no qty of cars, no year built. Assumed there's no garage here at all
variaveis_quanti.loc[pos_garage] = variaveis_quanti.loc[pos_garage].fillna(0)  
#!
# Some garages have no built year, but they have area (so, they exist). It's safe to assume they were made along with the house.
filtro = variaveis_quanti[variaveis_quanti['GarageYrBlt'].isnull()].index.values
variaveis_quanti.loc[filtro, 'GarageYrBlt'] = variaveis_quanti.loc[filtro, 'YearBuilt']
#!
# We should check the built years, to see if everything is fine
# (I've aleady checked max YrSold, it's 2010)
variaveis_quanti[variaveis_quanti['YearBuilt'] > variaveis_quanti['YrSold'].max()]
variaveis_quanti[variaveis_quanti['YearRemodAdd'] > variaveis_quanti['YrSold'].max()]
variaveis_quanti[variaveis_quanti['GarageYrBlt'] > variaveis_quanti['YrSold'].max()]  # DING! We've got a record error here
#!
year_error = variaveis_quanti[variaveis_quanti['GarageYrBlt'] > variaveis_quanti['YrSold'].max()].index.values  # let's get its index value
variaveis_quanti.loc[year_error, 'GarageYrBlt'] # yeap, record informs '2207' as the year, let's just assume it's '2007', the remodel year
variaveis_quanti.loc[year_error, 'GarageYrBlt'] = variaveis_quanti.loc[year_error, 'YearRemodAdd']
#!
# There are a few garages that have built year before the house building year. Let's assume the house building year is the right one.
ajuste_ano = variaveis_quanti[variaveis_quanti['YearBuilt'] > variaveis_quanti['GarageYrBlt']].index.values
variaveis_quanti.loc[ajuste_ano, 'GarageYrBlt'] = variaveis_quanti.loc[ajuste_ano, 'YearBuilt']
#!
# A few obs have YrSold < YearBuilt|YearRemodAdd, adjusting to the greater value
mask = variaveis_quanti[(variaveis_quanti['YrSold'] < variaveis_quanti['YearBuilt']) | (variaveis_quanti['YrSold'] < variaveis_quanti['YearRemodAdd'])].index.values
#variaveis_quanti.loc[mask, ['YrSold', 'YearBuilt', 'YearRemodAdd']]
variaveis_quanti.loc[mask, 'YrSold'] = variaveis_quanti.loc[mask, 'YearRemodAdd']
#!
# LotFrontage and LotArea have 0.489896 correlation overall, I'm using LotArea infomation to estimate LotFrontage values
# Just uncomment the line below to check the correlations if you want
# variaveis_quanti.drop('Origem', axis=1)[variaveis_quanti['LotFrontage'].isnull() == False].corr()['LotFrontage']
#!
# For the LotFrontage feature, I'll first take the mean value from the existing observations, working with
# each dataset independently in order to avoid data leakage
#!
# We'll gather existing LotFrontage information, calculate its ratio against LotArea and the mean of this ratios. 
# With this mean we'll estimate missing LotFrontage obs, for train and test sets independently
existing_frontage = np.where(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino']['LotFrontage'].isnull() == False)[0]
ratio_treino = variaveis_quanti.loc[existing_frontage, 'LotFrontage'] / variaveis_quanti.loc[existing_frontage, 'LotArea'] 
ratio_mean = ratio_treino.mean()
missing_idx = np.where(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino']['LotFrontage'].isnull())[0]
missing_frontage = variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'].loc[missing_idx]['LotArea']*ratio_mean
variaveis_quanti.loc[missing_frontage.index.values, 'LotFrontage'] = missing_frontage
#!
# For the test data we'll make a copy beforehand, then we'll do the same thing as above
test_data = variaveis_quanti[variaveis_quanti['Origem'] == 'Teste'].reset_index(drop=True).copy()
existing_frontage = np.where(test_data['LotFrontage'].isnull() == False)[0]
ratio_teste = test_data.loc[existing_frontage, 'LotFrontage'] / test_data.loc[existing_frontage, 'LotArea'] 
ratio_mean = ratio_teste.mean()
missing_idx_teste = np.where(test_data['LotFrontage'].isnull())[0]
missing_frontage = test_data.loc[missing_idx_teste]['LotArea']*ratio_mean
test_data.loc[missing_frontage.index.values, 'LotFrontage'] = missing_frontage
#!
# After calculations is done, we reintegrate test_data into the original grouped dataset
variaveis_quanti.drop(variaveis_quanti[variaveis_quanti['Origem'] == 'Teste'].index.values, inplace=True)
variaveis_quanti = pd.concat([variaveis_quanti, test_data])
#!
# MasVnrArea with NaN info relates to all MasVnrType type 'None', so area is zero, you can prove it 
# through this piece of code down here (uncomment, of course)
#indices_quali = variaveis_quali[variaveis_quali['MasVnrType'] == 'None']['MasVnrType'].index.values
#variaveis_quanti.iloc[indices_quali]['MasVnrArea'].isnull().sum() == variaveis_quanti['MasVnrArea'].isnull().sum()  # 23
variaveis_quanti['MasVnrArea'] = variaveis_quanti['MasVnrArea'].fillna(0)
#!
# In here we just have to deal with 'Bsmt' features, you can check below all null values correspond to
# just two observations, Id 2121 (test idx 660) and Id 2189 (test idx 728)
"""
variaveis_quanti[variaveis_quanti['BsmtFinSF1'].isnull()]
variaveis_quanti[variaveis_quanti['BsmtFinSF2'].isnull()]
variaveis_quanti[variaveis_quanti['BsmtUnfSF'].isnull()]
variaveis_quanti[variaveis_quanti['TotalBsmtSF'].isnull()]
variaveis_quanti[variaveis_quanti['BsmtFullBath'].isnull()]
variaveis_quanti[variaveis_quanti['BsmtHalfBath'].isnull()]
# confirm through categorical variables
variaveis_quali[variaveis_quali['Id'] == 2121].T
variaveis_quali[variaveis_quali['Id'] == 2189].T
"""
variaveis_quanti = variaveis_quanti.fillna(0)
variaveis_quanti.reset_index(drop=True, inplace=True)
#!
# If you wanna checkout all numeric features with no missing values, run the code below (uncomment first)
#variaveis_quanti.isnull().sum()
#!
#!
#########################################################################################3
#!
#!
#####
# FEATURE ENGENEERING
#####
#!
# Here are some new features that I've created that helped my models to perform better
#!
#!
#### Ages...
# ...of the proprety...
# We'll create three new features regarding the property age at the moment of the sale.
abs_age = variaveis_quanti['YrSold'] - variaveis_quanti['YearBuilt']
variaveis_quanti['IdadeCasa'] = abs_age
#!
# ... of the remodelling...
remod_age = variaveis_quanti['YrSold'] - variaveis_quanti['YearRemodAdd']
variaveis_quanti['IdadeReforma'] = remod_age
#!
# ...and of the garage.
gar_age = variaveis_quanti['YrSold'] - variaveis_quanti['GarageYrBlt']
variaveis_quanti['GarAge'] = gar_age
#!
# All done, we're able to drop some of this already used features
variaveis_quanti.drop(['GarageYrBlt', 'YearBuilt', 'YearRemodAdd'], axis=1, inplace=True)
#!
#### MoSold (month that the property was sold)
# It's a categorical feature, not a numerical feature. Adjusting.
variaveis_quali['MoSold'] = variaveis_quanti['MoSold'].reset_index(drop=True)
variaveis_quali['MoSold'] = variaveis_quali['MoSold'].astype(object)
variaveis_quanti = variaveis_quanti.drop('MoSold', axis = 1)
#!
#### MSSubClass
# Another categorical feature (it's even in the name - 'class')
variaveis_quali['MSSubClass'] = variaveis_quanti['MSSubClass'].reset_index(drop=True)
variaveis_quali['MSSubClass'] = variaveis_quali['MSSubClass'].astype(object)
variaveis_quanti = variaveis_quanti.drop('MSSubClass', axis = 1)
#!
#### YrSold
# We used it to define the properties ages, now we turn it into categorical info.
variaveis_quali['YrSold'] = variaveis_quanti['YrSold'].reset_index(drop=True)
variaveis_quali['YrSold'] = variaveis_quali['YrSold'].astype(object)
variaveis_quanti = variaveis_quanti.drop('YrSold', axis = 1)
#!
#### Basement
# TotalBsmtSF is the sum of three other features, 'BsmtFinSF1', 'BsmtFinSF2' e 'BsmtUnfSF'.
# We'll create a new feature, 'PoraoUtil', with just the finished area from the basement
porao_util = variaveis_quanti['TotalBsmtSF'] - variaveis_quanti['BsmtUnfSF']
variaveis_quanti['PoraoUtil'] = porao_util
#!
# You may try tailoring a model not dropping the former basement
# features if you want, just comment the line below
variaveis_quanti = variaveis_quanti.drop(['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF'], axis = 1)
#!
#!
#### Areas features
#!
# Created features:
#- AndaresProntos = 1stFlrSF + 2ndFlrSF (the whole habitable area)
#- TerrenoLivre = LotArea - 1stFlrSF (total area of the land minus the area of the house)
#- AreaCasaTotal = AndaresProntos + basement - low quality finished square feet
andares_prontos = variaveis_quanti['2ndFlrSF'] + variaveis_quanti['1stFlrSF']
area_livre_terreno = variaveis_quanti['LotArea'] - variaveis_quanti['1stFlrSF']
area_casa_total = andares_prontos + porao_util - variaveis_quanti['LowQualFinSF']
variaveis_quanti['AndaresProntos'] = andares_prontos
variaveis_quanti['TerrenoLivre'] = area_livre_terreno
variaveis_quanti['AreaCasaTotal'] = area_casa_total
#!
#### Leisure area
# New feature ('AreaLazerTotal') with the total leisure area
variaveis_quanti['AreaLazerTotal'] =  variaveis_quanti['WoodDeckSF'] + variaveis_quanti['OpenPorchSF'] + \
        variaveis_quanti['EnclosedPorch'] + variaveis_quanti['3SsnPorch'] + \
        variaveis_quanti['ScreenPorch'] + variaveis_quanti['PoolArea']
#!
#!
#] Deck, pool and porch categorical features
#!
# Deck
deck = {'foo'}
deck = pd.DataFrame(deck)
#!
for i in variaveis_quanti['WoodDeckSF']:
    if i == 0: deck.loc[len(deck)] = 'None'
    else: deck.loc[len(deck)] = 'Yes'
#!
deck = deck.iloc[1:]
deck.reset_index(inplace=True, drop=True)
#!
# Porch
porch = variaveis_quanti['OpenPorchSF'] + variaveis_quanti['EnclosedPorch'] + variaveis_quanti['3SsnPorch'] + variaveis_quanti['ScreenPorch']
varanda = {'foo'}
varanda = pd.DataFrame(varanda)
#!
for i in porch:
    if i == 0: varanda.loc[len(varanda)] = 'None'
    else: varanda.loc[len(varanda)] = 'Yes'
#!
varanda = varanda.iloc[1:]
varanda.reset_index(inplace=True, drop=True)
#!
# Pool
piscina = {'foo'}
piscina = pd.DataFrame(piscina)
#!
for i in variaveis_quanti['PoolArea']:
    if i == 0: piscina.loc[len(piscina)] = 'None'
    else: piscina.loc[len(piscina)] = 'Yes'
#!
piscina = piscina.iloc[1:]
piscina.reset_index(inplace=True, drop=True)
#!
#!
# Directing categorical and numerical features
variaveis_quali['Piscina'] = piscina
variaveis_quali['Varanda'] = varanda
variaveis_quali['Deck'] = deck
variaveis_quanti['PorchTotal'] = porch
#!
#!
#### Bathrooms
house_baths = variaveis_quanti['FullBath'] + variaveis_quanti['HalfBath']
basement_baths = variaveis_quanti['BsmtFullBath'] + variaveis_quanti['BsmtHalfBath']
total_baths = house_baths + basement_baths
variaveis_quanti['BanheirosCasa'] = house_baths
variaveis_quanti['BanheirosPorao'] = basement_baths
variaveis_quanti['BanheirosTotal'] = total_baths
variaveis_quanti.drop(['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath'], axis=1, inplace=True)
#!
#########################################################################################3
#!
#####
# CHECKING FOR OUTLIERS
#####
# IMPORTANT: check outliers ONLY in train sets, never in test sets. One cannot drop any observation in test sets, ever.
# I really like boxplot graphs for this. Of course that I'd alrady gone through a bunch of features,
#!
out_features = ['LotArea', 'SalePrice', 'MasVnrArea', '1stFlrSF', 'GrLivArea', 'LowQualFinSF', 'LotFrontage']
#!
"""
plt.figure(figsize =(10, 8))
sns.boxplot(data = variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'].drop(out_features, axis=1),
            saturation = 0.8,
            fill = False,
            width = 0.3,
            gap = 0.3,
            whis = 1.5, # IQR
            linecolor = 'auto',
            linewidth = 1,
            fliersize = None,
            native_scale = False)
plt.title('Boxplot das variáveis quanti', loc='center', fontsize=14)
plt.ylabel('Valores')
plt.show()
"""
#!
# Estimating interquartile ranges (IQR)
# Lower IQ
q1_lotarea = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[0]], 25)
q1_saleprice = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[1]], 25)
q1_masvnr = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[2]], 25)
q1_1stflr = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[3]], 25)
q1_grliv = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[4]], 25)
q1_lowqual = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[5]], 25)
q1_lotfront = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[6]], 25)
#!
# Upper IQ
q3_lotarea = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[0]], 75)
q3_saleprice = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[1]], 75)
q3_masvnr = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[2]], 75)
q3_1stflr = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[3]], 75)
q3_grliv = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[4]], 75)
q3_lowqual = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[5]], 75)
q3_lotfront = np.percentile(variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'][out_features[6]], 75)
#!
# IQR
iqr_lotarea = q3_lotarea - q1_lotarea
iqr_saleprice = q3_saleprice - q1_saleprice
iqr_masvnr = q3_lotarea - q1_lotarea
iqr_1stflr = q3_lotarea - q1_lotarea
iqr_grliv = q3_lotarea - q1_lotarea
iqr_lowqual = q3_lotarea - q1_lotarea
iqr_lotfront = q3_lotarea - q1_lotarea
#!
# Estimating just the upper limit, as almost no feature has lower outliers
outliers_lotarea_sup = q3_lotarea + 1.5 * iqr_lotarea
outliers_saleprice_sup = q3_saleprice + 1.5 * iqr_saleprice
outliers_masvnr_sup = q3_masvnr + 1.5 * iqr_masvnr
outliers_1stflr_sup = q3_1stflr + 1.5 * iqr_1stflr
outliers_grliv_sup = q3_grliv + 1.5 * iqr_grliv
outliers_lowqual_sup = q3_lowqual + 1.5 * iqr_lowqual
outliers_lotfront_sup = q3_lotfront + 1.5 * iqr_lotfront
#!
# Creates a mask (idx_outliers) so I can easily drop the outliers, if it is needed
mask = variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'].index
mask = ((variaveis_quanti.loc[mask, out_features[0]] >= outliers_lotarea_sup) | (variaveis_quanti.loc[mask, out_features[1]] >= outliers_saleprice_sup) |
        (variaveis_quanti.loc[mask, out_features[2]] >= outliers_masvnr_sup) | (variaveis_quanti.loc[mask, out_features[3]] >= outliers_1stflr_sup) |
        (variaveis_quanti.loc[mask, out_features[4]] >= outliers_grliv_sup) | (variaveis_quanti.loc[mask, out_features[5]] >= outliers_lowqual_sup) |
        (variaveis_quanti.loc[mask, out_features[6]] >= outliers_lotfront_sup))
#!
idx_outliers = variaveis_quanti[variaveis_quanti['Origem'] == 'Treino'].loc[mask].index
#!
#variaveis_quanti.drop(idx_outliers, inplace=True)
#!
#!
#####
# PRE-TRAINING ADJUSTMENTS: ONE HOT ENCODING
#####

# As is well known, one hot encoding is only applicable in categorical features. I like to
# use sklearn package to do so.
#
# Calls OHE and processes it, dropping the first column from every original categorical feature
ohe = OneHotEncoder(categories = 'auto', drop = 'first', sparse_output = False).set_output(transform = 'pandas')
variaveis_ohe = ohe.fit_transform(variaveis_quali.drop(['Origem', 'Id'], axis=1))
#!
# Now we create our two big datasets, one with OHE and one without, already dropping columns
# 'Id' that are of no more use for our work now
dados_ohe = pd.concat([variaveis_quanti.reset_index(drop=True), variaveis_ohe], axis = 1).drop('Id', axis=1)  # one hot encoded data
dados_sem = pd.concat([variaveis_quanti.reset_index(drop=True), variaveis_quali.drop('Origem', axis=1).astype('category')], axis = 1).drop('Id', axis=1)  # No hot encoding
#!
# Just a little information here:
# 'Curse of Dimensionality' refers to the challenges in analyzing data with many variables (in absolute terms or in relation to 
# the number of observations).
# - The number of samples needed to estimate a function grows exponentially with the number of variables. 
# - The volume of space increases quickly as dimensions are added, making data sparse. 
# - The complexity of the problem increases with the number of variables. 
#!
# Let's split our datasets back to train/test sets
ohe_test = dados_ohe[dados_ohe['Origem'] == 'Teste'].reset_index(drop=True).drop(['SalePrice', 'Origem'], axis=1)
ohe_train = dados_ohe[dados_ohe['Origem'] == 'Treino'].reset_index(drop=True).drop('Origem', axis=1)
#!
noohe_test = dados_sem[dados_sem['Origem'] == 'Teste'].reset_index(drop=True).drop(['SalePrice', 'Origem'], axis=1)
noohe_train = dados_sem[dados_sem['Origem'] == 'Treino'].reset_index(drop=True).drop('Origem', axis=1)
#!
#!
#########################################################################################3
#!
#####
# MODELS AREA
#####
#!
#!
# Defining target and also train size (%)
target = ['SalePrice']
tamanho_treino = 0.8
#!
# ** OHE data train/test information **
train_data_ohe = ohe_train.drop(target, axis = 1)
test_data_ohe = ohe_train[target]
# OHE data train/test split
train_x_ohe, test_x_ohe, train_y_ohe, test_y_ohe = train_test_split(train_data_ohe, test_data_ohe, train_size = tamanho_treino, random_state = 1)
#!#!
# ** Non-OHE data train/test split **
train_data = noohe_train.drop(target, axis = 1)
test_data = noohe_train[target]
# Non-OHE data train/test split
train_x, test_x, train_y, test_y = train_test_split(train_data, test_data, train_size = tamanho_treino, random_state = 1)
#!
#!
# NOTE: the '*' simbol besides the score information in comments means that this specific
# score resulted from previous submissions, not the one with RMSLE, R2 and CV infos
# recorded




#### Neural Network Model
#!
# Criação da função de loss
def rmsle(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return tf.keras.ops.sqrt(msle(y_true, y_pred)) 


# OHE Model (local RMSLE 0.439562, R2 0.786799)
normalizador = keras.layers.Normalization(axis=-1)
normalizador.adapt(np.array(test_x_ohe))

# Criação da rede
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo_dnn = keras.Sequential()
#modelo_dnn = keras.Sequential([normalizador])
modelo_dnn.add(keras.layers.Dense(64, activation='relu', name='Input_Dense64'))
modelo_dnn.add(keras.layers.Dense(64, activation='relu', name='Hidden1_Dense64'))
modelo_dnn.add(keras.layers.Dropout(0.3, seed=1))
modelo_dnn.add(keras.layers.Dense(128, activation='relu', name='HiddenL3_Dense128'))
modelo_dnn.add(keras.layers.Dense(1, name='Output_Dense1'))
modelo_dnn.compile(loss=rmsle, 
                   optimizer='Adam',
                   metrics=['mean_squared_logarithmic_error'])
#1
callback = keras.callbacks.EarlyStopping(monitor='loss',  # callback de early stopping
                                         patience=3)
#!
fitted = modelo_dnn.fit(train_x_ohe,
                        train_y_ohe,
                        epochs=100,
                        validation_data=(test_x_ohe, test_y_ohe),
                        batch_size=32,
                        callbacks=[callback])
#1
# Prediction analysis
ypred_dnn = modelo_dnn.predict(test_x_ohe)#.flatten()
rmsle_dnn = np.sqrt(modelo_dnn.evaluate(test_x_ohe, test_y_ohe, verbose=1))
metric = keras.metrics.R2Score()
metric.update_state(test_y_ohe, ypred_dnn)
score_dnn_r2 = metric.result()
print(f'\nLocal RMSLE: {rmsle_dnn[0]:.6f}')
print(f'Coefficient of Determination (R2): {score_dnn_r2:.6f}')

# Loss graph
plt.plot(fitted.history['loss'])
plt.plot(fitted.history['val_loss'])
plt.title(f'Loss do modelo {nome_modelo}\nDados de treino com One Hot Encoding')
plt.ylabel('RMSLE')
plt.xlabel(f'Iterações (epochs)\n\nScore final (RMSLE): {rmsle_dnn[0]:.6f}   R2: {score_dnn_r2:.6f}')
plt.legend(['Treino', 'Teste'])
plt.grid(True)
plt.show()

#### SUBMISSION
# Predicts the target on test data
ypred_dnn_final = modelo_dnn.predict(ohe_test)
# Generates the name to save file
print(f'Prediction in use has record #{nome_modelo}.')
print('Compiling filename to submission...')
ext_final = ('.csv')
save_final = ('TensorFlow_DNN_OHE_'+ str(round(rmsle_dnn[0], 5))+'_'+(nome_modelo + ext_final))
print('Generating submission file named ' + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv('sample_submission.csv')
submissao['SalePrice'] = ypred_dnn_final
submissao.to_csv('./resultados/'+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")










# Non-OHE Model (local RMSLE 0.999999, R2 0.000000)

# Layer de Categoricals
categ = keras.layers.CategoryEncoding(num_tokens=2, output_mode="one_hot")
feat_names = train_x.dtypes[train_x.dtypes == 'category'].index.values






normalizador = keras.layers.Normalization(axis=-1)
normalizador.adapt(np.array(test_x))




# Criação da rede
nome_modelo = datetime.now().strftime("%Y%m%d-%H%M")
modelo_dnn = keras.Sequential()
#modelo_dnn = keras.Sequential([normalizador])
modelo_dnn.add(keras.layers.Dense(128, activation='relu', name='Input_Dense128'))
modelo_dnn.add(keras.layers.Dense(128, activation='relu', name='Hidden1_Dense128'))
modelo_dnn.add(keras.layers.Dropout(0.2, seed=1))
modelo_dnn.add(keras.layers.Dense(128, activation='relu', name='HiddenL3_Dense128'))
modelo_dnn.add(keras.layers.Dense(1, name='Output_Dense1'))
modelo_dnn.compile(loss=rmsle,
                   optimizer='Adam',
                   metrics=['mean_squared_logarithmic_error'])
#1
callback = keras.callbacks.EarlyStopping(monitor='loss',  # callback de early stopping
                                         patience=3)
#!
fitted = modelo_dnn.fit(train_x,
                        train_y,
                        epochs=40,
                        validation_data=(test_x, test_y),
                        batch_size=32,
                        callbacks=[callback])

# Prediction analysis
ypred_dnn = modelo_dnn.predict(test_x)#.flatten()
rmsle_dnn = np.sqrt(modelo_dnn.evaluate(test_x, test_y, verbose=1))
metric = keras.metrics.R2Score()
metric.update_state(test_y, ypred_dnn)
score_dnn_r2 = metric.result()
print(f'\nLocal RMSLE: {rmsle_dnn[0]:.6f}')
print(f'Coefficient of Determination (R2): {score_dnn_r2:.6f}')

# Loss graph
plt.plot(fitted.history['loss'])
plt.plot(fitted.history['val_loss'])
plt.title(f'Loss do modelo {nome_modelo}\nDados de treino com One Hot Encoding')
plt.ylabel('RMSLE')
plt.xlabel(f'Iterações (epochs)\n\nScore final (RMSLE): {rmsle_dnn:.6f}   R2: {score_dnn_r2:.6f}')
plt.legend(['Treino', 'Teste'])
plt.grid(True)
plt.show()

#### SUBMISSION
# Predicts the target on test data
ypred_dnn_final = modelo_dnn.predict(noohe_test)
# Generates the name to save file
print(f'Prediction in use has record #{nome_modelo}.')
print('Compiling filename to submission...')
ext_final = ('.csv')
save_final = ('TensorFlow_DNN_'+ str(round(rmsle_dnn, 5))+'_'+(nome_modelo + ext_final))
print('Generating submission file named ' + save_final)
# Reads submission file, records the obtained data and rewrites the file with it
submissao = pd.read_csv('sample_submission.csv')
submissao['SalePrice'] = ypred_dnn_final
submissao.to_csv('./resultados/'+save_final, index=False)
print("\nSuccess. File is in directory ('./resultados/').")
