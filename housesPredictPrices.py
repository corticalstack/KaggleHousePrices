# Jon-Paul Boyd - Kaggle - House Value Prediction
# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet, LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Lasso

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

    
    # Outliers - quick analysis through the standard deviation of 'SalePrice' and a set of scatter plots.
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
    var = 'GrLivArea'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    
      
    # Bivariate analysis saleprice/grlivarea
    var = 'TotalBsmtSF'
    data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000))
    
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
            'BsmtQual': 'NA',
            'BsmtCond': 'NA',
            'BsmtExposure': 'NA',
            'BsmtFinType1': 'NA',
            'BsmtFinType2': 'NA',
            'BsmtFullBath': 0,
            'BsmtHalfBath': 0,
            'KitchenQual': 'TA',
            'Functional': 'Typ',
            'FireplaceQu': 'NA',
            'GarageType': 'NoGarage',
            'GarageFinish': 'NA',
            'GarageQual': 'NA',
            'GarageCond': 'NA',   
            'PoolQC': 'NA',
            'Fence': 'NoFence',
            'MiscFeature': 'None',
            'Electrical' : 'SBrkr'
            })

    df.loc[df.MasVnrType == 'None', 'MasVnrArea'] = 0
    df.loc[df.BsmtFinType1=='NA', 'BsmtFinSF1'] = 0
    df.loc[df.BsmtFinType2=='NA', 'BsmtFinSF2'] = 0
    df.loc[df.BsmtFinType1=='NA', 'BsmtUnfSF'] = 0
    df.loc[df.BsmtFinType1=='NA', 'TotalBsmtSF'] = 0
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
    # convert to strings 
    df['MSSubClass'] = df['MSSubClass'].apply(str)
    return df


def set_new_columns(df):
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_Home_Quality'] = df['OverallQual'] + df['OverallCond']    
    
    df['YearBuiltBin'] = pd.qcut(df['YearBuilt'], 10)
    label = LabelEncoder()
    df['YearBuiltBin_Code'] = label.fit_transform(df['YearBuiltBin'])
    df.drop(['YearBuiltBin'], axis = 1, inplace = True)

    df['YearRemodAddBin'] = pd.qcut(df['YearRemodAdd'], 5)
    label = LabelEncoder()
    df['YearRemodAddBin_Code'] = label.fit_transform(df['YearRemodAddBin'])
    df.drop(['YearRemodAddBin'], axis = 1, inplace = True)

    df.drop(['YearBuilt'], axis = 1, inplace = True)
    df.drop(['MoSold'], axis = 1, inplace = True)
    df.drop(['YrSold'], axis = 1, inplace = True)
    return df


def log1p(df):
    t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
         'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 
         'GrLivArea', 'GarageArea', 'PoolArea', 'MiscVal', 
         'OverallQual', 'OverallCond', 'TotalSF']
    
    df.loc[:, t] = np.log1p(df.loc[:, t])    
    return df


def ordinal_cats(df):
    cat_qual_1 = ['Ex', 'Gd', 'TA', 'Fa', 'Po', 'NA']
    cat_qual_2 = ['Gd', 'Av', 'Mn', 'No', 'NA']
    cat_qual_3 = ['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA']
    cat_qual_4 = ['Gtl', 'Mod', 'Sev']
    cat_qual_5 = ['Gravel', 'Paved', 'NA']
    cat_qual_6 = ['Y', 'P', 'N']
    cat_qual_7 = ['Fin', 'RFn', 'Unf', 'NA']
    cat_qual_8 = ['Y', 'N']
    
    df['ExterQual'] = pd.Categorical(df['ExterQual'], categories=cat_qual_1, ordered=True).codes
    df['ExterCond'] = pd.Categorical(df['ExterCond'], categories=cat_qual_1, ordered=True).codes
    df['BsmtQual'] = pd.Categorical(df['BsmtQual'], categories=cat_qual_1, ordered=True).codes
    df['BsmtCond'] = pd.Categorical(df['BsmtCond'], categories=cat_qual_1, ordered=True).codes
    df['BsmtExposure'] = pd.Categorical(df['BsmtExposure'], categories=cat_qual_2, ordered=True).codes
    df['BsmtFinType1'] = pd.Categorical(df['BsmtFinType1'], categories=cat_qual_3, ordered=True).codes
    df['BsmtFinType2'] = pd.Categorical(df['BsmtFinType2'], categories=cat_qual_3, ordered=True).codes
    df['HeatingQC'] = pd.Categorical(df['HeatingQC'], categories=cat_qual_1, ordered=True).codes
    df['KitchenQual'] = pd.Categorical(df['KitchenQual'], categories=cat_qual_1, ordered=True).codes
    df['FireplaceQu'] = pd.Categorical(df['FireplaceQu'], categories=cat_qual_1, ordered=True).codes
    df['GarageQual'] = pd.Categorical(df['GarageQual'], categories=cat_qual_1, ordered=True).codes
    df['GarageCond'] = pd.Categorical(df['GarageCond'], categories=cat_qual_1, ordered=True).codes
    df['GarageFinish'] = pd.Categorical(df['GarageFinish'], categories=cat_qual_7, ordered=True).codes
    df['PoolQC'] = pd.Categorical(df['PoolQC'], categories=cat_qual_1, ordered=True).codes
    df['LandSlope'] = pd.Categorical(df['LandSlope'], categories=cat_qual_4, ordered=True).codes
    df['Street'] = pd.Categorical(df['Street'], categories=cat_qual_5, ordered=True).codes
    df['Alley'] = pd.Categorical(df['Alley'], categories=cat_qual_4, ordered=True).codes
    df['PavedDrive'] = pd.Categorical(df['PavedDrive'], categories=cat_qual_6, ordered=True).codes
    df['CentralAir'] = pd.Categorical(df['CentralAir'], categories=cat_qual_8, ordered=True).codes
    
    return df


def get_dummies(df):
    df = pd.get_dummies(df, columns=['MSSubClass', 'MSZoning', 'LotShape', 
       'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 
       'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 
       'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 
       'Heating', 'Electrical', 'Functional', 'GarageType', 
       'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'], drop_first=True)
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


def addSquared(df):
    columns_list = ['LotFrontage', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                'GarageCars', 'GarageArea', 'OverallQual','ExterQual','BsmtQual',
                'GarageQual','FireplaceQu','KitchenQual', 'TotalSF']
    m = df.shape[1]
    for c in columns_list:
        df = df.assign(newcol=pd.Series(df[c]*df[c]).values)   
        df.columns.values[m] = c + '_sq'
        m += 1
    return df 


df_train = handle_missing(df_train)
df_train = ordinal_cats(df_train)
df_train = set_new_columns(df_train)
df_train = change_types(df_train)

df_test = handle_missing(df_test)
df_test = ordinal_cats(df_test)
df_test = set_new_columns(df_test)
df_test = change_types(df_test)

# Square columns
df_train = addSquared(df_train)
df_test = addSquared(df_test)

# Remove several outliers
df_train.drop(df_train[(df_train['OverallQual']<5) & (df_train['SalePrice']>200000)].index, inplace=True)
df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index, inplace=True)
df_train.reset_index(drop=True, inplace=True)

# Log
df_train = log1p(df_train)
df_train['SalePrice'] = np.log1p(df_train['SalePrice'])
df_test = log1p(df_test)

# Convert categorical variables into dummies 
df_train = get_dummies(df_train)
df_test = get_dummies(df_test)
df_test = dummies_missing_cols(df_train, df_test)

y = df_train.SalePrice
df_train.drop(['SalePrice'], axis=1, inplace=True)

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005, 5e-4], max_iter=5000)
model_lasso.fit(df_train, y)

coef = pd.Series(model_lasso.coef_, index = df_train.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = pd.concat([coef.sort_values().head(20),
                     coef.sort_values().tail(20)])
    
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

# Convert train to numpy array and delete index column
X = np.array(df_train)
X = np.delete(X, 0, axis=1)

test_errors_regr_lasso = []
test_errors_regr_ridge = []
test_errors_regr_gbr = []
test_errors_regr_enet = []
test_errors_regr_lasso_stacked = []

nFolds = 20
ifold = 1
models = []

kf = KFold(n_splits=nFolds, random_state=241, shuffle=True)

for train_index, test_index in kf.split(X):
    print('fold: ',ifold)
    ifold = ifold + 1    
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # lasso
    regr_lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0003, random_state=1, max_iter=50000))
    regr_lasso.fit(X_train, y_train)
    regr_lasso_train_pred = regr_lasso.predict(X_train)
    regr_lasso_test_pred = regr_lasso.predict(X_test)


    # Ridge
    regr_ridge = Ridge(alpha=9.0, fit_intercept = True)
    regr_ridge.fit(X_train, y_train)
    regr_ridge_train_pred = regr_ridge.predict(X_train)
    regr_ridge_test_pred = regr_ridge.predict(X_test)


    # Gradient Boosting    
    regr_gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.02,
                                          max_depth=4, max_features='sqrt',
                                          min_samples_leaf=15, min_samples_split=50,
                                          loss='huber', random_state = 5)         
    regr_gbr.fit(X_train, y_train)
    regr_gbr_train_pred = regr_gbr.predict(X_train)
    regr_gbr_test_pred = regr_gbr.predict(X_test)
        
       
    # Elastic Net
    regr_enet = make_pipeline(RobustScaler(), ElasticNet(alpha=4.0, l1_ratio=0.005, random_state=3))
    regr_enet.fit(X_train, y_train)
    regr_enet_train_pred = regr_enet.predict(X_train) 
    regr_enet_test_pred = regr_enet.predict(X_test) 
    
        
    # Stacking
    stacked_set = pd.DataFrame({'A' : []})
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_lasso_test_pred)], axis=1)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_ridge_test_pred)], axis=1)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_gbr_test_pred)], axis=1) 
    stacked_set = pd.concat([stacked_set, pd.DataFrame(regr_enet_test_pred)], axis=1)
    product = (regr_lasso_test_pred*regr_ridge_test_pred*regr_gbr_test_pred*regr_enet_test_pred) ** (1.0/4.0)
    stacked_set = pd.concat([stacked_set, pd.DataFrame(product)], axis=1)
    Xstack = np.array(stacked_set)
    Xstack = np.delete(Xstack, 0, axis=1)
    regr_lasso_stacked = Lasso(alpha = 0.0001,fit_intercept = True)
    regr_lasso_stacked.fit(Xstack, y_test)
    regr_lasso_stacked_Xstack_pred = regr_lasso_stacked.predict(Xstack)
    
    models.append([regr_ridge, regr_lasso, regr_gbr, regr_enet, regr_lasso_stacked])
    
    test_errors_regr_lasso.append(np.square(regr_lasso_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_ridge.append(np.square(regr_ridge_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_gbr.append(np.square(regr_gbr_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_enet.append(np.square(regr_enet_test_pred - y_test).mean() ** 0.5)
    test_errors_regr_lasso_stacked.append(np.square(regr_lasso_stacked_Xstack_pred - y_test).mean() ** 0.5)


print('Lasso test error: ', np.mean(test_errors_regr_lasso))
print('Ridge test error: ', np.mean(test_errors_regr_ridge))
print('Gradient Boosting test error: ', np.mean(test_errors_regr_gbr))
print('Elastic Net test error: ', np.mean(test_errors_regr_enet))
print('Lasso stacked test error: ', np.mean(test_errors_regr_lasso_stacked))


# Convert test to numpy array and delete index column
X_score = np.array(df_test)
X_score = np.delete(X_score, 0, axis=1)
M = X_score.shape[0]
scores_final = 1+np.zeros(M)

for model in models:
    model_lasso = model[0]
    model_ridge = model[1]
    model_gbr = model[2]
    model_enet = model[3]
    model_lasso_stacked = model[4]
    
    model_lasso_scores = model_lasso.predict(X_score)
    model_ridge_scores = model_ridge.predict(X_score)
    model_gbr_scores = model_gbr.predict(X_score)
    model_enet_scores = model_enet.predict(X_score)
    
    stacked_sets = pd.DataFrame({'A' : []})
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_lasso_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_ridge_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_gbr_scores)],axis=1)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(model_enet_scores)],axis=1)
    
    product = (model_lasso_scores*model_ridge_scores*model_gbr_scores*model_enet_scores) ** (1.0/4.0)
    stacked_sets = pd.concat([stacked_sets, pd.DataFrame(product)], axis=1)    
    Xstacks = np.array(stacked_sets)
    Xstacks = np.delete(Xstacks, 0, axis=1)
    scores_final = scores_final * model_lasso_stacked.predict(Xstacks)


scores_final = scores_final ** (1/nFolds)

Id = df_test['Id']
fin_score = pd.DataFrame({'SalePrice': np.exp(scores_final)-1})
fin_data = pd.concat([Id,fin_score],axis=1)
    

# Brutal approach to deal with predictions close to outer range 
q1 = fin_data['SalePrice'].quantile(0.0042)
q2 = fin_data['SalePrice'].quantile(0.99)
fin_data['SalePrice'] = fin_data['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
fin_data['SalePrice'] = fin_data['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

# Output    
fin_data.to_csv('test_set_prediction.csv', sep=',', index = False)
      