# Jon-Paul Boyd - Kaggle - Classifier to Predict Titantic Survial 
# Importing the libraries
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns
import re as re
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
#from hyperopt_gbc import Hyperopt_gbc

#processDataAnalysis=True
processDataAnalysis=False

# Importing the datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df_full = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'], df_test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)


if processDataAnalysis:
    # Distribution Analysis on training set
    print('_'*80, "=== Training set info ===", sep='\n')
    print(df_train.columns.values, '_'*80, sep='\n')
    print(df_train.info(), '_'*80, sep='\n')
    print(df_train.head(), '_'*80, sep='\n')
    print(df_train.tail(), '_'*80, sep='\n')
    print(df_train.describe(), '_'*80, sep='\n')
    print(df_train.describe(include=['O']), '_'*80, sep='\n')
    
    # Distribution analysis of target variable SalePrice
    df_train['SalePrice'].describe()
    sns.distplot(df_train['SalePrice']);
    print("Skewness: %f" % df_train['SalePrice'].skew())
    print("Kurtosis: %f" % df_train['SalePrice'].kurt())
    
    # Scatter and box plots with SalePrice and other interger variables
    # OverallQual: Rates the overall material and finish of the house
    var = 'OverallQual'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    
    # OverallCond: Rates the overall condition of the house
    var = 'OverallCond'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    
    # YearBuilt: Original construction date
    var = 'YearBuilt'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    f, ax = plt.subplots(figsize=(16, 8))
    fig = sns.boxplot(x=var, y="SalePrice", data=data)
    fig.axis(ymin=0, ymax=800000)
    plt.xticks(rotation=90)
    
    # BsmtFinSF1: Type 1 finished square feet
    var = 'BsmtFinSF1'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Basement Type 1 Finished Sq Ft')
    
    # BsmtFinSF2: Type 2 finished square feet
    var = 'BsmtFinSF2'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000),  title='Basement Type 2 Finished Sq Ft')
    
    # TotalBsmtSF: Total square feet of basement area
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Basement Total Sq Ft')
    
    # 1stFlrSF: First Floor square feet
    var = '1stFlrSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='First Floor Sq Ft')
    
    # 2ndFlrSF: Second floor square feet
    var = '2ndFlrSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Second Floor Sq Ft')
    
    # LowQualFinSF: Low quality finished square feet (all floors)
    var = 'LowQualFinSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Low Quality Sq Ft')
    
    # GrLivArea: Above grade (ground) living area square feet
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Above Ground Living Sq Ft')
    
    # Bedroom: Bedrooms above grade (does NOT include basement bedrooms)
    var = 'BedroomAbvGr'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Bedrooms Above Grade')
    
    # TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    var = 'TotRmsAbvGrd'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Total Rooms Above Grade')
    
    # GarageArea: Size of garage in square feet
    var = 'GarageArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Garage Sq Ft')
    
    # GarageCars: Size of garage in car capacity
    var = 'GarageCars'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Garage Sq Ft')
    
    # PoolArea: Pool area in square feet
    var = 'PoolArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000), title='Pool Sq Ft')
    
    # Correlation matrix (heatmap style)
    corrmat = df_train.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=.8, square=True)


    # Saleprice correlation matrix - top K largest using pandas corr
    # Look for the ligh coloured squares indicating highly correlated variables with
    # SalePrice such as OverallQual, GrLivArea, then TotalBsmtSF, GarageArea etc
    # OverallQual', 'GrLivArea' and 'TotalBsmtSF' are strongly correlated with 'SalePrice'. 
    # 'GarageCars' and 'GarageArea' are also some of the most strongly correlated variables. 
    # Just need one of these variables in our analysis (we can keep 'GarageCars' since its correlation with 'SalePrice' is higher).
    # 'YearBuilt' is slightly correlated with 'SalePrice'
    k = 10 #number of variables for heatmap
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    
    # Scatter Pair plot
    # One of the figures we may find interesting is the one between 'TotalBsmtSF' and 'GrLiveArea'. 
    # In this figure we can see the dots drawing a linear line, which almost acts like a border. 
    # It totally makes sense that the majority of the dots stay below that line. Basement areas 
    # can be equal to the above ground living area, but it is not expected a basement area bigger 
    # than the above ground living area (unless you're trying to buy a bunker).
    
    # The plot concerning 'SalePrice' and 'YearBuilt' can also make us think. In the bottom of 
    # the 'dots cloud', we see what almost appears to be a shy exponential function (be creative). 
    # We can also see this same tendency in the upper limit of the 'dots cloud' (be even more creative). 
    # Also, notice how the set of dots regarding the last years tend to stay above this limit 
    # (I just wanted to say that prices are increasing faster now).
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(df_train[cols], size = 2.5)
    plt.show()
    
    #Outliers - quick analysis through the standard deviation of 'SalePrice' and a set of scatter plots.
    # Standardizing data - How 'SalePrice' looks. Low range values are similar and not too far from 0.
    # High range values are far from 0 and the 7.something values are really out of range.
    # For now, we'll not consider any of these values as an outlier but we should be careful with those two 7.something values.
    saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis])
    low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
    high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
    print('outer range (low) of the distribution:')
    print(low_range)
    print('\nouter range (high) of the distribution:')
    print(high_range)
    
    # Missing data - train
    # How prevalent is the missing data?
    # Is missing data random or does it have a pattern?
    total = df_train.isnull().sum().sort_values(ascending=False)
    percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    missing_data.head(20)

 # Bivariate analysis saleprice/grlivarea
    # The two values with bigger 'GrLivArea' seem strange and they are not following the crowd. 
    # We can speculate why this is happening. Maybe they refer to agricultural area and that 
    # could explain the low price. I'm not sure about this but I'm quite confident that these 
    # two points are not representative of the typical case. Therefore, we'll define them as 
    # outliers and delete them.
    
    # The two observations in the top of the plot are those 7.something observations that we 
    # said we should be careful about. They look like two special cases, however they seem 
    # to be following the trend. For that reason, we will keep them.
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    
    
    #deleting points
    #df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
    #df_train = df_train.drop(df_train[dataset_full['Id'] == 1299].index)
    #df_train = df_train.drop(df_train[dataset_full['Id'] == 524].index)
    
    # Bivariate analysis saleprice/grlivarea
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    
    #df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
    #df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
    #df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
    
    
    # In search of normality - histogram and normal probability plot
    # Normal probability plot - Data distribution should closely follow the diagonal 
    # that represents the normal distribution.
    sns.distplot(df_train['SalePrice'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_train['SalePrice'], plot=plt)


    df_full.groupby('MSZoning').count()
    df_full.groupby('Alley').count()
    df_full.groupby('Utilities').count()
    df_full.groupby('Exterior1st').count()
    df_full.groupby('Exterior2nd').count()
    df_full.groupby('MasVnrType').count()
    df_full.groupby('BsmtQual').count()
    df_full.groupby('BsmtCond').count()
    df_full.groupby('BsmtExposure').count()
    df_full.groupby('BsmtFinType1').count()
    df_full.groupby('BsmtFinType2').count()
    df_full.groupby('BsmtFullBath').count()
    df_full.groupby('KitchenQual').count()
    df_full.groupby('Functional').count()
    df_full.groupby('FireplaceQu').count()
    df_full.groupby('GarageType').count()
    df_full.groupby('GarageFinish').count()
    df_full.groupby('GarageQual').count()
    df_full.groupby('GarageCond').count()
    df_full.groupby('PoolQC').count()
    df_full.groupby('Fence').count()
    df_full.groupby('MiscFeature').count()
    df_full.groupby('Electrical').count()
    df_full.groupby('SaleType').count()
    
def handle_missing(df):
    
    df = df.fillna({
            'MSZoning': 'RL',        
            'Alley': 'NoAlley',
            'Utilities': 'AllPub',
            'Exterior1st': 'VinylSd',
            'Exterior2nd': 'VinylSd',
            'MasVnrType': 'None',
            'BsmtQual': 'NoBsmt',
            'BsmtCond': 'NoBsmt',
            'BsmtExposure': 'NoBsmt',
            'BsmtFinType1': 'NoBsmt',
            'BsmtFinType2': 'NoBsmt',
            'BsmtFullBath': 0,
            'BsmtHalfBath': 0,
            'KitchenQual': 'TA',
            'Functional': 'Typ',
            'FireplaceQu': 'NoFireplace',
            'GarageType': 'NoGarage',
            'GarageFinish': 'NoGarage',
            'GarageQual': 'NoGarage',
            'GarageCond': 'NoGarage',   
            'PoolQC': 'NoPool',
            'Fence': 'NoFence',
            'MiscFeature': 'None',
            'Electrical' : 'SBrkr'
            })

    df.loc[df.MasVnrType == 'None', 'MasVnrArea'] = 0
    df.loc[df.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
    df.loc[df.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
    df.loc[df.BsmtFinType1=='NoBsmt', 'BsmtUnfSF'] = 0
    df.loc[df.BsmtFinType1=='NoBsmt', 'TotalBsmtSF'] = 0
    df.loc[df['GarageYrBlt'].isnull(), 'GarageYrBlt'] = df['YearBuilt']
    df.loc[df['GarageArea'].isnull(), 'GarageArea'] = df.loc[df['GarageType']=='Detchd', 'GarageArea'].mean()
    df.loc[df['GarageCars'].isnull(), 'GarageCars'] = df.loc[df['GarageType']=='Detchd', 'GarageCars'].median()
    
    x = df.loc[np.logical_not(df["LotFrontage"].isnull()), "LotArea"]
    y = df.loc[np.logical_not(df["LotFrontage"].isnull()), "LotFrontage"]
    t = (x <= 25000) & (y <= 150)
    p = np.polyfit(x[t], y[t], 1)
    df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, df.loc[df['LotFrontage'].isnull(), 'LotArea'])
    
    return df


def change_types(df):
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    df['YrSold'] = df['YrSold'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)
    return df


def log1p(df):
    t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
     '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
     'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    df.loc[:, t] = np.log1p(df.loc[:, t])    
    return df


def get_dummies(df):
    df = pd.get_dummies(df, columns=['MSZoning', 'Street',
       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 
       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
       'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
       'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
       'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
       'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], drop_first=True)
    return df


def dummies_missing_cols(train, test):
    # Get missing columns in the training test
    missing_cols = set(train.columns ) - set(test.columns)
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        test[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    test = test[train.columns]
    test.drop(['SalePrice'], axis=1, inplace=True)
    test.reset_index(drop=True, inplace=True)
    return test

    
df_train = handle_missing(df_train)
df_train = change_types(df_train)

df_test = handle_missing(df_test)
df_test = change_types(df_test)

# Applying log transformation
#train['SalePrice'] = np.log(train['SalePrice'])
#dataset_full['SalePrice'] = np.log(dataset_full['SalePrice'])

# Transformed histogram and normal probability plot
#sns.distplot(df_train['SalePrice'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['SalePrice'], plot=plt)

# Histogram and normal probability plot
#sns.distplot(df_train['GrLivArea'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['GrLivArea'], plot=plt)

# Data transformation
#df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#dataset_full['GrLivArea'] = np.log(dataset_full['GrLivArea'])

# Transformed histogram and normal probability plot
#sns.distplot(df_train['GrLivArea'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['GrLivArea'], plot=plt)

# Histogram and normal probability plot
#sns.distplot(df_train['TotalBsmtSF'], fit=norm)
#fig = plt.figure()
#res = stats.probplot(df_train['TotalBsmtSF'], plot=plt)

   


# Scatter plot
#plt.scatter(df_train['GrLivArea'], df_train['SalePrice'])
# As suggested by many participants, we remove several outliers
df_train.drop(df_train[(df_train['OverallQual']<5) & (df_train['SalePrice']>200000)].index, inplace=True)
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)
df_train.reset_index(drop=True, inplace=True)



#df_full.BsmtFinSF1.clip(df_full.BsmtFinSF1.mean() - (3*df_full.BsmtFinSF1.std()), df_full.BsmtFinSF1.mean() + (3*df_full.BsmtFinSF1.std()), inplace=True)
#df_full.BsmtFinSF2.clip(df_full.BsmtFinSF2.mean() - (3*df_full.BsmtFinSF2.std()), df_full.BsmtFinSF2.mean() + (3*df_full.BsmtFinSF2.std()), inplace=True)
#df_full.TotalBsmtSF.clip(df_full.TotalBsmtSF.mean() - (3*df_full.TotalBsmtSF.std()), df_full.TotalBsmtSF.mean() + (3*df_full.TotalBsmtSF.std()), inplace=True)
#df_full.LowQualFinSF.clip(df_full.LowQualFinSF.mean() - (3*df_full.LowQualFinSF.std()), df_full.LowQualFinSF.mean() + (3*df_full.LowQualFinSF.std()), inplace=True)
#df_full.GrLivArea.clip(df_full.GrLivArea.mean() - (3*df_full.GrLivArea.std()), df_full.GrLivArea.mean() + (3*df_full.GrLivArea.std()), inplace=True)
#df_full.GarageArea.clip(df_full.GarageArea.mean() - (3*df_full.GarageArea.std()), df_full.GarageArea.mean() + (3*df_full.GarageArea.std()), inplace=True)
#df_full.BsmtFinSF1.clip(df_full.BsmtFinSF1.mean() - (3*df_full.BsmtFinSF1.std()), df_full.BsmtFinSF1.mean() + (3*df_full.BsmtFinSF1.std()), inplace=True)
#df_full.BsmtFinSF1.clip(df_full.BsmtFinSF1.mean() - (3*df_full.BsmtFinSF1.std()), df_full.BsmtFinSF1.mean() + (3*df_full.BsmtFinSF1.std()), inplace=True)


df_train = log1p(df_train)
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
df_test = log1p(df_test)


# Convert categorical variables into dummies 
df_train = get_dummies(df_train)
df_test = get_dummies(df_test)

df_test = dummies_missing_cols(df_train, df_test)

from sklearn.grid_search import ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

y = df_train.SalePrice
df_train.drop(['SalePrice'], axis=1, inplace=True)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4], max_iter=70000).fit(df_train, y)

coef = pd.Series(model_lasso.coef_, index = df_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
    
plt.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Lasso Model")
plt.show()
    
#let's look at the residuals as well:
plt.rcParams['figure.figsize'] = (6.0, 6.0)

preds = pd.DataFrame({"preds":model_lasso.predict(df_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x = "preds", y = "residuals",kind = "scatter")
plt.show()
    
lasso_preds = np.expm1(model_lasso.predict(df_test))


solution = pd.DataFrame({"id":df_test.Id, "SalePrice":lasso_preds})
solution.to_csv("test_set_prediction.csv", index = False)
      