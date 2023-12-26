import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, make_scorer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import time

# Import data
data_path = r"C:\Users\sonaa\OneDrive\Рабочий стол\predictive_maintenance.csv"
data = pd.read_csv(data_path)

# Basic checks
print('Data summary:')
data.info()
print('Duplicate values present:', data['Product ID'].nunique() != len(data))
# Set numeric columns dtype to float
data['Tool wear [min]'] = data['Tool wear [min]'].astype('float64')
data['Rotational speed [rpm]'] = data['Rotational speed [rpm]'].astype('float64')
# Rename features
data.rename(mapper={'Air temperature [K]': 'Air temperature',
                    'Process temperature [K]': 'Process temperature',
                    'Rotational speed [rpm]': 'Rotational speed',
                    'Torque [Nm]': 'Torque',
                    'Tool wear [min]': 'Tool wear'}, axis=1, inplace=True)
# Import data
data_path = r"/predictive_maintenance.csv"
data = pd.read_csv(data_path)

# Basic checks
print('Data summary:')
data.info()
print('Duplicate values present:', data['Product ID'].nunique() != len(data))
# Remove first character and set to numeric dtype
data['Product ID'] = data['Product ID'].apply(lambda x: x[1:])
data['Product ID'] = pd.to_numeric(data['Product ID'])

# Histogram of ProductID
sns.histplot(data=data, x='Product ID', hue='Type')
plt.show()
# Drop ID columns
df = data.copy()
df.drop(columns=['UDI', 'Product ID'], inplace=True)
# Pie chart of Type percentage
value = data['Type'].value_counts()
Type_percentage = 100 * value / data.Type.shape[0]
labels = Type_percentage.index.array
x = Type_percentage.array
plt.pie(x, labels=labels, colors=sns.color_palette('tab10')[0:3], autopct='%.0f%%')
plt.title('Machine Type percentage')
plt.show()
# Create lists of features and target names
features = [col for col in df.columns
            if df[col].dtype == 'float64' or col == 'Type']
target = ['Target', 'Failure Type']
# Portion of data where RNF=1
idx_RNF = df.loc[df['Failure Type'] == 'Random Failures'].index
print(df.loc[idx_RNF, target])

first_drop = df.loc[idx_RNF, target].shape[0]
print('Number of observations where RNF=1 but Machine failure=0:', first_drop)
# Drop corresponding observations and RNF column
print(df.drop(index=idx_RNF, inplace=True))
# Portion of data where Machine failure=1 but no failure cause is specified
idx_ambiguous = df.loc[(df['Target'] == 1) &
                       (df['Failure Type'] == 'No Failure')].index
second_drop = df.loc[idx_ambiguous].shape[0]
print('Number of ambiguous observations:', second_drop)
print(df.loc[idx_ambiguous, target])
df.drop(index=idx_ambiguous, inplace=True)
# Global percentage of removed observations
n = df.shape[0]
print('Global percentage of removed observations:',
      (100 * (first_drop + second_drop) / n))
df.reset_index(drop=True, inplace=True)
print(df.describe())
num_features = [feature for feature in features if df[feature].dtype == 'float64']
# Histograms of numeric features
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Numeric features histogram')
for j, feature in enumerate(num_features):
    sns.histplot(ax=axs[j // 3, j - 3 * (j // 3)], data=df, x=feature)
plt.show()

# boxplot of numeric features
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Numeric features boxplot')
for j, feature in enumerate(num_features):
    sns.boxplot(ax=axs[j // 3, j - 3 * (j // 3)], data=df, x=feature)
plt.show()
# Portion of df where there is a failure and causes percentage
idx_fail = df.loc[df['Failure Type'] != 'No Failure'].index
df_fail = df.loc[idx_fail]
df_fail_percentage = 100 * df_fail['Failure Type'].value_counts() / df_fail['Failure Type'].shape[0]
print('Failures percentage in data:',
      round(100 * df['Target'].sum() / n, 2))
# Pie plot
plt.title('Causes involved in Machine failures')
plt.pie(x=df_fail_percentage.array, labels=df_fail_percentage.index.array,
        colors=sns.color_palette('tab10')[0:4], autopct='%.0f%%')
plt.show()
# n_working must represent 80% of the desired length of resampled dataframe
n_working = df['Failure Type'].value_counts()['No Failure']
desired_length = round(n_working / 0.8)
spc = round((desired_length - n_working) / 4)  # samples per class
# Resampling
balance_cause = {'No Failure': n_working,
                 'Overstrain Failure': spc,
                 'Heat Dissipation Failure': spc,
                 'Power Failure': spc,
                 'Tool Wear Failure': spc}
sm = SMOTENC(categorical_features=[0, 7], sampling_strategy=balance_cause, random_state=0)
df_res, y_res = sm.fit_resample(df, df['Failure Type'])
# Portion of df_res where there is a failure and causes percentage
idx_fail_res = df_res.loc[df_res['Failure Type'] != 'No Failure'].index
df_res_fail = df_res.loc[idx_fail_res]
fail_res_percentage = 100 * df_res_fail['Failure Type'].value_counts() / df_res_fail.shape[0]

# Percentages
print('Percentage increment of observations after oversampling:',
      round((df_res.shape[0] - df.shape[0]) * 100 / df.shape[0], 2))
print('SMOTE Resampled Failures percentage:',
      round(df_res_fail.shape[0] * 100 / df_res.shape[0], 2))

# Pie plot
fig, axs = plt.subplots(ncols=2, figsize=(12, 4))
fig.suptitle('Causes involved in Machine failures')
axs[0].pie(x=df_fail_percentage.array, labels=df_fail_percentage.index.array,
           colors=sns.color_palette('tab10')[0:4], autopct='%.0f%%')
axs[1].pie(x=fail_res_percentage.array, labels=fail_res_percentage.index.array,
           colors=sns.color_palette('tab10')[0:4], autopct='%.0f%%')
axs[0].title.set_text('Original')
axs[1].title.set_text('After Resampling')
plt.show()
# Kdeplot of numeric features (After resampling) - hue=Type
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(19, 7))
fig.suptitle('Features distribution (After resampling)')
custom_palette = {'L': 'tab:blue', 'M': 'tab:orange', 'H': 'tab:green'}
for j, feature in enumerate(num_features):
    sns.kdeplot(ax=axs[j // 3, j - 3 * (j // 3)], data=df_res, x=feature,
                hue='Type', fill=True, palette=custom_palette)
plt.show()
# Kdeplot of numeric features (Original)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Original Features distribution')
enumerate_features = enumerate(num_features)
for j, feature in enumerate_features:
    sns.kdeplot(ax=axs[j // 3, j - 3 * (j // 3)], data=df, x=feature,
                hue='Target', fill=True, palette='tab10')
plt.show()
# Kdeplot of numeric features (After resampling)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Features distribution after oversampling')
enumerate_features = enumerate(num_features)
for j, feature in enumerate_features:
    sns.kdeplot(ax=axs[j // 3, j - 3 * (j // 3)], data=df_res, x=feature,
                hue=df_res['Target'], fill=True, palette='tab10')
plt.show()
# Kdeplot of numeric features (After resampling) - Diving deeper
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 7))
fig.suptitle('Features distribution after oversampling - Diving deeper')
enumerate_features = enumerate(num_features)
for j, feature in enumerate_features:
    sns.kdeplot(ax=axs[j // 3, j - 3 * (j // 3)], data=df_res, x=feature,
                hue=df_res['Failure Type'], fill=True, palette='tab10')
plt.show()
sc = StandardScaler()
type_dict = {'L': 0, 'M': 1, 'H': 2}
cause_dict = {'No Failure': 0,
              'Power Failure': 1,
              'Overstrain Failure': 2,
              'Heat Dissipation Failure': 3,
              'Tool Wear Failure': 4}
df_pre = df_res.copy()
# Encoding
df_pre['Type'].replace(to_replace=type_dict, inplace=True)
df_pre['Failure Type'].replace(to_replace=cause_dict, inplace=True)
# Scaling
df_pre[num_features] = sc.fit_transform(df_pre[num_features])
pca = PCA(n_components=len(num_features))
X_pca = pd.DataFrame(data=pca.fit_transform(df_pre[num_features]),
                     columns=['PC' + str(i + 1) for i in range(len(num_features))])
var_exp = pd.Series(data=100 * pca.explained_variance_ratio_,
                    index=['PC' + str(i + 1) for i in range(len(num_features))])
print('The variance ratio explained by each component: ', round(var_exp, 2), sep='\n')
print('The variance ratio explained by the three components: ' + str(round(var_exp.values[:3].sum(), 2)))
# PCA for Data visualization
pca3 = PCA(n_components=3)
X_pca3 = pd.DataFrame(data=pca3.fit_transform(df_pre[num_features]), columns=['PC1', 'PC2', 'PC3'])

# Loadings Analysis
fig, axs = plt.subplots(ncols=3, figsize=(18, 4))
fig.suptitle('Loadings magnitude')
pca_loadings = pd.DataFrame(data=pca3.components_, columns=num_features)
for j in range(3):
    ax = axs[j]
    sns.barplot(ax=ax, x=pca_loadings.columns, y=pca_loadings.values[j])
    ax.tick_params(axis='x', rotation=90)
    ax.title.set_text('PC' + str(j + 1))
plt.show()
X_pca3.rename(mapper={'PC1': 'Temperature',
                      'PC2': 'Power',
                      'PC3': 'Tool Wear'}, axis=1, inplace=True)

# PCA plot
color = []
col = df_pre['Failure Type'].map({0: 'tab:blue', 1: 'tab:orange', 2: 'tab:green', 3: 'tab:red', 4: 'tab:purple'})
color.append(col)
idx_w = col[col == 'tab:blue'].index
color.append(col.drop(idx_w))
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
labelTups = [('No Failure', 'tab:blue'),
             ('Power Failure', 'tab:orange'),
             ('Overstrain Failure', 'tab:green'),
             ('Heat Dissipation Failure', 'tab:red'),
             ('Tool Wear Failure', 'tab:purple')]

fig = plt.figure(figsize=(18, 6))
fig.suptitle('Data in 3D PCA space')
full_idx = X_pca3.index

for j, idx in enumerate([full_idx, idx_fail_res]):
    ax = fig.add_subplot(1, 2, j + 1, projection='3d')

    lg = ax.scatter(X_pca3.loc[idx, 'Temperature'],
                    X_pca3.loc[idx, 'Power'],
                    X_pca3.loc[idx, 'Tool Wear'],
                    c=color[j])
    ax.set_xlabel('$Temperature$')
    ax.set_ylabel('$Power$')
    ax.set_zlabel('$Tool Wear$')
    ax.title.set_text('With' + str(j * 'out') + ' "No Failure" class')
    ax.view_init(35, -10)
    custom_lines = [plt.Line2D([], [], ls="", marker='.',
                               mec='k', mfc=c, mew=.1, ms=20) for c in colors[j:]]
    ax.legend(custom_lines, [lt[0] for lt in labelTups[j:]],
              loc='center left', bbox_to_anchor=(1.0, .5))

plt.show()
# Correlation Heatmap
plt.figure(figsize=(7, 4))
sns.heatmap(data=df_pre.corr(), mask=np.triu(df_pre.corr()), annot=True, cmap='BrBG')
plt.title('Correlation Heatmap')
plt.show()
# train-validation-test split
X, y = df_pre[features], df_pre[['Target', 'Failure Type']]
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, stratify=df_pre['Failure Type'],
                                                          random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.11,
                                                  stratify=y_trainval['Failure Type'], random_state=0)
def eval_preds(model, X, y_true, y_pred, task):
    if task == 'binary':
        # Extract task target
        y_true = y_true['Target']
        cm = confusion_matrix(y_true, y_pred)
        # Probability of the minority class
        proba = model.predict_proba(X)[:, 1]
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba)
        f1 = f1_score(y_true, y_pred, pos_label=1)
        f2 = fbeta_score(y_true, y_pred, pos_label=1, beta=2)
    elif task == 'multi_class':
        y_true = y_true['Failure Type']
        cm = confusion_matrix(y_true, y_pred)
        proba = model.predict_proba(X)
        # Metrics
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, proba, multi_class='ovr', average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        f2 = fbeta_score(y_true, y_pred, beta=2, average='weighted')
    metrics = pd.Series(data={'ACC': acc, 'AUC': auc, 'F1': f1, 'F2': f2})
    metrics = round(metrics, 3)
    return cm, metrics
def tune_and_fit(clf, X, y, params, task):
    if task == 'binary':
        f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,
                                  cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Target'])
    elif task == 'multi_class':
        f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
        start_time = time.time()
        grid_model = GridSearchCV(clf, param_grid=params,
                                  cv=5, scoring=f2_scorer)
        grid_model.fit(X, y['Failure Type'])

    print('Best params:', grid_model.best_params_)
    # Print training times
    train_time = time.time() - start_time
    mins = int(train_time // 60)
    print('Training time: ' + str(mins) + 'm ' + str(round(train_time - mins * 60)) + 's')
    return grid_model


def predict_and_evaluate(fitted_models, X, y_true, clf_str, task):
    cm_dict = {key: np.nan for key in clf_str}
    metrics = pd.DataFrame(columns=clf_str)
    y_pred = pd.DataFrame(columns=clf_str)
    for fit_model, model_name in zip(fitted_models, clf_str):
        # Update predictions
        y_pred[model_name] = fit_model.predict(X)
        # Metrics
        if task == 'binary':
            cm, scores = eval_preds(fit_model, X, y_true,
                                    y_pred[model_name], task)
        elif task == 'multi_class':
            cm, scores = eval_preds(fit_model, X, y_true,
                                    y_pred[model_name], task)
        # Update Confusion matrix and metrics
        cm_dict[model_name] = cm
        metrics[model_name] = scores
    return y_pred, cm_dict, metrics


def fit_models(clf, clf_str, X_train, X_val, y_train, y_val):
    metrics = pd.DataFrame(columns=clf_str)
    for model, model_name in zip(clf, clf_str):
        model.fit(X_train, y_train['Target'])
        y_val_pred = model.predict(X_val)
        metrics[model_name] = eval_preds(model, X_val, y_val, y_val_pred, 'binary')[1]
    return metrics


# Models
lr = LogisticRegression()
knn = KNeighborsClassifier()
svc = SVC(probability=True)
rfc = RandomForestClassifier()
xgb = XGBClassifier()

clf = [lr, knn, svc, rfc, xgb]
clf_str = ['LR', 'KNN', 'SVC', 'RFC', 'XGB']

# Fit on raw train
metrics_0 = fit_models(clf, clf_str, X_train, X_val, y_train, y_val)

# Fit on temperature product train
XX_train = X_train.drop(columns=['Process temperature', 'Air temperature'])
XX_val = X_val.drop(columns=['Process temperature', 'Air temperature'])
XX_train['Temperature'] = X_train['Process temperature'] * X_train['Air temperature']
XX_val['Temperature'] = X_val['Process temperature'] * X_val['Air temperature']
metrics_1 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# Fit on power product train
XX_train = X_train.drop(columns=['Rotational speed', 'Torque'])
XX_val = X_val.drop(columns=['Rotational speed', 'Torque'])
XX_train['Power'] = X_train['Rotational speed'] * X_train['Torque']
XX_val['Power'] = X_val['Rotational speed'] * X_val['Torque']
metrics_2 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# Fit on both products train
XX_train = X_train.drop(columns=['Process temperature', 'Air temperature', 'Rotational speed', 'Torque'])
XX_val = X_val.drop(columns=['Process temperature', 'Air temperature', 'Rotational speed', 'Torque'])
XX_train['Temperature'] = X_train['Process temperature'] * X_train['Air temperature']
XX_val['Temperature'] = X_val['Process temperature'] * X_val['Air temperature']
XX_train['Power'] = X_train['Rotational speed'] * X_train['Torque']
XX_val['Power'] = X_val['Rotational speed'] * X_val['Torque']
metrics_3 = fit_models(clf, clf_str, XX_train, XX_val, y_train, y_val)

# classification metrics barplot
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
fig.suptitle('Classification metrics')
for j, model in enumerate(clf_str):
    ax = axs[j // 3, j - 3 * (j // 3)]
    model_metrics = pd.DataFrame(data=[metrics_0[model], metrics_1[model], metrics_2[model], metrics_3[model]])
    model_metrics.index = ['Original', 'Temperature', 'Power', 'Both']
    model_metrics.transpose().plot(ax=ax, kind='bar', rot=0, )
    ax.title.set_text(model)
    ax.get_legend().remove()
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
axs.flatten()[-2].legend(title='Dataset', loc='upper center',
                         bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=12)
plt.show()
# Make predictions
lr = LogisticRegression(random_state=0)
lr.fit(X_train, y_train['Target'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)

# Metrics
cm_val_lr, metrics_val_lr = eval_preds(lr, X_val, y_val, y_val_lr, 'binary')
cm_test_lr, metrics_test_lr = eval_preds(lr, X_test, y_test, y_test_lr, 'binary')
print('Validation set metrics:', metrics_val_lr, sep='\n')
print('Test set metrics:', metrics_test_lr, sep='\n')

cm_labels = ['Not Failure', 'Failure']
cm_lr = [cm_val_lr, cm_test_lr]
# Show Confusion Matrices
fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
fig.suptitle('LR Confusion Matrices')
for j, title in enumerate(['Validation Set', 'Test Set']):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_lr[j], annot=True,
                fmt='d', cmap='Blues', cbar=False)
    axs[j].title.set_text(title)
    axs[j].set_xticklabels(cm_labels)
    axs[j].set_yticklabels(cm_labels)
plt.show()

# Odds for interpretation
d = {'feature': X_train.columns, 'odds': np.exp(lr.coef_[0])}
odds_df = pd.DataFrame(data=d).sort_values(by='odds', ascending=False)
odds_df
# Models
knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
xgb = XGBClassifier()
clf = [knn,svc,rfc,xgb]
clf_str = ['KNN','SVC','RFC','XGB']

# Parameter grids for GridSearch
knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[300,500,700],
              'max_depth':[5,7],
              'learning_rate':[0.01,0.1],
              'objective':['binary:logistic']}
params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params],
                   index=clf)

# Tune hyperparameters with GridSearch (estimated time 8m)
print('GridSearch start')
fitted_models_binary = []
for model, model_name in zip(clf, clf_str):
    print('Training '+str(model_name))
    fit_model = tune_and_fit(model,X_train,y_train,params[model],'binary')
    fitted_models_binary.append(fit_model)
    # Create evaluation metrics
    task = 'binary'
    y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(
        fitted_models_binary, X_val, y_val, clf_str, task)
    y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(
        fitted_models_binary, X_test, y_test, clf_str, task)

    # Show Validation Confusion Matrices
    fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
    fig.suptitle('Validation Set Confusion Matrices')
    for j, model_name in enumerate(clf_str):
        ax = axs[j]
        sns.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True,
                    fmt='d', cmap='Blues', cbar=False)
        ax.title.set_text(model_name)
        ax.set_xticklabels(cm_labels)
        ax.set_yticklabels(cm_labels)
    plt.show()

    # Show Test Confusion Matrices
    fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
    fig.suptitle('Test Set Confusion Matrices')
    for j, model_name in enumerate(clf_str):
        ax = axs[j]
        sns.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True,
                    fmt='d', cmap='Blues', cbar=False)
        ax.title.set_text(model_name)
        ax.set_xticklabels(cm_labels)
        ax.set_yticklabels(cm_labels)
    plt.show()

    # Print scores
    print('')
    print('Validation scores:', metrics_val, sep='\n')
    print('Test scores:', metrics_test, sep='\n')
    # Evaluate Permutation Feature Importances
    f2_scorer = make_scorer(fbeta_score, pos_label=1, beta=2)
    importances = pd.DataFrame()
    for clf in fitted_models_binary:
        result = permutation_importance(clf, X_train, y_train['Target'],
                                        scoring=f2_scorer, random_state=0)
        result_mean = pd.Series(data=result.importances_mean, index=X.columns)
        importances = pd.concat(objs=[importances, result_mean], axis=1)
    importances.columns = clf_str

    # Barplot of Feature Importances
    fig, axs = plt.subplots(ncols=4, figsize=(20, 4))
    fig.suptitle('Permutation Feature Importances')
    for j, name in enumerate(importances.columns):
        sns.barplot(ax=axs[j], x=importances.index, y=importances[name].values)
        axs[j].tick_params('x', labelrotation=90)
        axs[j].set_ylabel('Importances')
        axs[j].title.set_text(str(name))
    plt.show()
# multiclass classification
lr = LogisticRegression(random_state=0,multi_class='ovr')
lr.fit(X_train, y_train['Failure Type'])
y_val_lr = lr.predict(X_val)
y_test_lr = lr.predict(X_test)

# Validation metrics
cm_val_lr, metrics_val_lr = eval_preds(lr,X_val,y_val,y_val_lr,'multi_class')
cm_test_lr, metrics_test_lr = eval_preds(lr,X_test,y_test,y_test_lr,'multi_class')
print('Validation set metrics:',metrics_val_lr, sep='\n')
print('Test set metrics:',metrics_test_lr, sep='\n')

cm_lr = [cm_val_lr, cm_test_lr]
cm_labels = ['No Fail','PWF','OSF','HDF','TWF']
# Show Confusion Matrices
fig, axs = plt.subplots(ncols=2, figsize=(9,4))
fig.suptitle('LR Confusion Matrices')
for j, title in enumerate(['Validation Set', 'Test Set']):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_lr[j], annot=True,
              fmt='d', cmap='Blues', cbar=False)
    axs[j].title.set_text(title)
    axs[j].set_xticklabels(cm_labels)
    axs[j].set_yticklabels(cm_labels)
plt.show()

# Odds for interpretation
odds_df = pd.DataFrame(data = np.exp(lr.coef_), columns = X_train.columns,
                       index = df_res['Failure Type'].unique())
print(odds_df)
# Models
knn = KNeighborsClassifier()
svc = SVC(decision_function_shape='ovr')
rfc = RandomForestClassifier()
xgb = XGBClassifier()
clf = [knn,svc,rfc,xgb]
clf_str = ['KNN','SVC','RFC','XGB']

knn_params = {'n_neighbors':[1,3,5,8,10]}
svc_params = {'C': [1, 10, 100],
              'gamma': [0.1,1],
              'kernel': ['rbf'],
              'probability':[True],
              'random_state':[0]}
rfc_params = {'n_estimators':[100,300,500,700],
              'max_depth':[5,7,10],
              'random_state':[0]}
xgb_params = {'n_estimators':[100,300,500],
              'max_depth':[5,7,10],
              'learning_rate':[0.01,0.1],
              'objective':['multi:softprob']}

params = pd.Series(data=[knn_params,svc_params,rfc_params,xgb_params],
                    index=clf)


# Tune hyperparameters with GridSearch (estimated time 8-10m)
print('GridSearch start')
fitted_models_multi = []
for model, model_name in zip(clf, clf_str):
    print('Training '+str(model_name))
    fit_model = tune_and_fit(model,X_train,y_train,params[model],'multi_class')
    fitted_models_multi.append(fit_model)
# Create evaluation metrics

task = 'multi_class'
y_pred_val, cm_dict_val, metrics_val = predict_and_evaluate(
    fitted_models_multi,X_val,y_val,clf_str,task)
y_pred_test, cm_dict_test, metrics_test = predict_and_evaluate(
    fitted_models_multi,X_test,y_test,clf_str,task)

# Show Validation Confusion Matrices
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('Validation Set Confusion Matrices')
for j, model_name in enumerate(clf_str):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_dict_val[model_name], annot=True,
                fmt='d', cmap='Blues', cbar=False)
    ax.title.set_text(model_name)
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)
plt.show()

# Show Test Confusion Matrices
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('Test Set Confusion Matrices')
for j, model_name in enumerate(clf_str):
    ax = axs[j]
    sns.heatmap(ax=ax, data=cm_dict_test[model_name], annot=True,
                fmt='d', cmap='Blues', cbar=False)
    ax.title.set_text(model_name)
    ax.set_xticklabels(cm_labels)
    ax.set_yticklabels(cm_labels)
plt.show()

# Print scores
print('')
print('Validation scores:', metrics_val, sep='\n')
print('Test scores:', metrics_test, sep='\n')
# Evaluate Permutation Feature Importances
f2_scorer = make_scorer(fbeta_score, beta=2, average='weighted')
importances = pd.DataFrame()
for clf in fitted_models_multi:
    result = permutation_importance(clf, X_train,y_train['Failure Type'],
                                  scoring=f2_scorer,random_state=0)
    result_mean = pd.Series(data=result.importances_mean, index=X.columns)
    importances = pd.concat(objs=[importances,result_mean],axis=1)

importances.columns = clf_str

# Barplot of Feature Importances
fig, axs = plt.subplots(ncols=4, figsize=(20,4))
fig.suptitle('Permutation Feature Importances')
for j, name in enumerate(importances.columns):
    sns.barplot(ax=axs[j], x=importances.index, y=importances[name].values)
    axs[j].tick_params('x',labelrotation=90)
    axs[j].set_ylabel('Importances')
    axs[j].title.set_text(str(name))
plt.show()
# Random Forest Decision Path
from sklearn import tree
import graphviz

tree_binary = fitted_models_binary[2].best_estimator_.estimators_[0]
tree_multi = fitted_models_multi[2].best_estimator_.estimators_[0]
trees = [tree_binary, tree_multi]
targets = ['Target', 'Failure Type']
for decision_tree, target in zip(trees, targets):
    decision_tree.fit(X_train, y_train[target])
    classes = list(map(str, df_res[target].unique()))

    dot_data = tree.export_graphviz(decision_tree, out_file=None,
                                  feature_names=X.columns,
                                  class_names=classes,
                                  filled=True, rounded=True,
                                  special_characters=True,
                                  max_depth=4)  # uncomment to see full tree
    graph = graphviz.Source(dot_data)
    graph.render(target+" Classification tree")
print(graph)
