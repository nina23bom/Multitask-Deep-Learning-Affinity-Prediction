import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics, model_selection

import gpm
import gpk

def score(Y, pred_Y, pred_var):
    r1 = stats.rankdata(Y)
    r2 = stats.rankdata(pred_Y)
    scores = {}
    scores['kendalltau'] = stats.kendalltau(r1, r2).correlation
    scores['R2'] = metrics.r2_score(Y, pred_Y)
    scores['MSE'] = metrics.mean_squared_error(Y, pred_Y)
    scores['R'] = np.corrcoef(Y, pred_Y)[0, 1]
    log_ps = -0.5 * np.log(pred_var) - (pred_Y - Y)**2 / 2 / pred_var
    log_ps -= 0.5 * np.log(2 * np.pi)
    scores['log_loss'] = -np.sum(log_ps)
    return scores

def select_ligand_sequence_X_and_Y(df, all_X, y_col, sequence_id):
    """
    Extract ys from df and Xs from embedding

    Input:
    df: original dataset 
    all_X : embedding
    y_column: output label (numerical for regression task)
    ----------
    Output:
    Xs: embedding 
    Ys: output label
    """
    df_ligand = pd.DataFrame(df['morganfp'].apply(lambda x: [int(i) for i in x]).tolist())
    df_ligand.index=df.index
    df_e_X = df.join(all_X, on=sequence_id)
    df_e_X = df_e_X[all_X.columns]   
    Xs = df_ligand.merge(df_e_X, left_index=True, right_index=True, how='inner', suffixes=('_bit', '_e'))
    Ys = df[y_col]
    return Xs, Ys
    
def cross_validate_ligand_sequence(df, select_ligand_sequence_X_and_Y, embeddings):
    column_names = ['kendalltau', 'R2', 'MSE','R','log_loss']
    embedding_cv = pd.DataFrame(index=embeddings, columns = column_names)
    for embedding in embeddings:
        #print(f"Evaluating {embedding}, index is {embeddings.index(embedding)}")
        with open(embedding_dir+embedding, 'rb') as f:
            e_X = pickle.load(f)
            if len(e_X) == 2:
                e_X = e_X[0]
        X, y = select_ligand_sequence_X_and_Y(df, e_X, y_col, sequence_id)
        X = X.values
        y = y.values

        #cross-validation predictions
        kf = model_selection.KFold(n_splits=10, shuffle=True, random_state=10)

        y_actual = []
        mu_test = []
        var_test = []
        mu_val = []
        var_val = []
        for i_train, i_val in kf.split(X):
            X_= X[i_train]
            y_ = y[i_train]
            X_val = X[i_val]
            y_val = y[i_val]
            y_actual.append(y_val)
            k = gpk.MaternKernel("5/2")
            clf = gpm.GPRegressor(k, gueses=(10,100))
            clf.fit(X_, y_)
            mu, var = clf.predict(X_val)
            mu_test.append(mu)
            var_test.append(np.diag(var))
        y_actual = np.concatenate(y_actual)
        mu_val = np.concatenate(mu_test)
        var_val = np.concatenate(var_test)
        embedding_cv.loc[embedding]= score(y_actual, mu_val, var_val)
    print(f"cv for {dataset} embedding completed")
    return embedding_cv

#######

### Load dataset for the regression model
dataset = "AD_R"
y_col = "mean_pChEMBL_Value"
sequence_id = "target_id"
dataset_dir = "/ccs/proj/chm194/nina23bom/Proj_4/doc2vec/ChEMBL_Datasets/"+dataset+"_MTL_ligand_sequence_random_split.csv"
df = pd.read_csv(dataset_dir)
### Load the doc2vec protein embedding models to be evaluated
embedding_dir = "/lustre/orion/chm194/scratch/nina23bom/Proj_4/doc2vec/docvec_outputs/"+dataset+"_"+"embeddings/"
embeddings = os.listdir(embedding_dir)
print(f"Total of {len(embeddings)} doc2vec embeddings to be evaluated")

start_time = time.time()

n_sample = 500
cv_df_list = []
for i in range(1,11):
    fold = "is_train_fold_"+str(i)
    print(fold)
    df_fold = df[df[fold]].iloc[0:n_sample,:]
    cv_df_list.append(cross_validate_ligand_sequence(df_fold, select_ligand_sequence_X_and_Y,embeddings))
print(f"Total of 20-fold cross validation completed for {dataset}")

end_time = time.time()
# Output the time
print(f"Time taken: {end_time - start_time} secnds")

cv_df_agg = pd.concat(cv_df_list,axis=1, keys=range(len(cv_df_list))).groupby(level=1, axis=1).mean()
sorted_cv_df_agg = cv_df_agg.sort_values(by=["MSE","kendalltau","log_loss"], ascending=[True, False, True]).reset_index(drop=False)

#print out a part of the sorted_cv_df_agg
print(sorted_cv_df_agg.head(5))
sorted_cv_df_agg.to_csv("/ccs/proj/chm194/nina23bom/Proj_4/doc2vec/docvec_outputs/"+dataset+"_random_split_cv.csv", index=False)
#sorted_cv_df_agg.to_csv("/gpfs/alpine/chm194/scratch/nina23bom/Proj_4/doc2vec/docvec_outputs/"+dataset+"_random_split_cv.csv",index=False)
