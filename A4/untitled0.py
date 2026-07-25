import pandas as pd
import numpy as np

def mahalanobis_dist(mat1, mat2, mat3, cinv):
    maha = np.zeros(len(mat1))
    for i in range(len(mat1)):
        xminusmeu_mat = np.array([mat1[i], mat2[i], mat3[i]])
        xminusmeu_mat_transpose = np.transpose(xminusmeu_mat)
        result = np.matmul(np.matmul(xminusmeu_mat_transpose, cinv), xminusmeu_mat)
        maha[i] = result
    return np.sqrt(maha)
def fncov(var1, var2):
    sum = 0
    length = len(var1)
    for i in range(length):
        sum = sum + (var1[i] * var2[i])
    return sum/(length-1)

tc = [4,6,5,7]
f  = [7,5,9,11]
avg = [10,8,6,9]

tcn = tc - np.mean(tc)
fn = f - np.mean(f)
avgn = avg - np.mean(avg)


uni_gram_df = pd.DataFrame({"gram_count":tcn,"document_count":fn, "Average_freq": avgn})
l1  = uni_gram_df["gram_count"].tolist()
l2 = uni_gram_df["document_count"].tolist()
l3 = uni_gram_df["Average_freq"].tolist()

covar = [[0] * 3 for i in range(3)]
lselect = [l1, l2, l3]

for i in range(3):
    for j in range(3):
        covar[i][j] = fncov(lselect[i], lselect[j])
        
covar_inv = np.linalg.inv(covar)

maha = mahalanobis_dist(l1, l2, l3, covar_inv)
print(maha)


def remove_univariate_outliers(df,
                               cols=('gram_count', 'document_count', 'Average_freq'),
                               z_thresh=3, mad_thresh=3, iqr_factor=1.5):
    df = df.copy()
    overall_flag = pd.Series(False, index=df.index)

    for col in cols:
        x = df[col]

        # 1) z-score rule
        z = (x - x.mean()) / x.std(ddof=0)
        z_flag = z.abs() > z_thresh

        # 2) IQR rule
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_low, iqr_high = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        iqr_flag = (x < iqr_low) | (x > iqr_high)

        # 3) MAD rule 
        med = x.median()
        mad = np.median(np.abs(x - med)) or 1  # avoid division by zero
        mad_z = 0.6745 * (x - med) / mad
        mad_flag = mad_z.abs() > mad_thresh

        # update master flag (logical OR across rules and columns)
        overall_flag |= z_flag | iqr_flag | mad_flag

    return df.loc[~overall_flag]


def keep_univariate_outliers(df,
                             cols=('gram_count', 'document_count', 'Average_freq'),
                             z_thresh=3, mad_thresh=3, iqr_factor=1.5):
    df = df.copy()
    overall_out = pd.Series(False, index=df.index)

    for col in cols:
        x = df[col]

        # ── 1) z-score rule ─────────────────────────
        z = (x - x.mean()) / x.std(ddof=0)
        z_flag = z.abs() > z_thresh

        # ── 2) IQR rule  ──────── ─────
        q1, q3 = x.quantile([0.25, 0.75])
        iqr     = q3 - q1
        low, hi = q1 - iqr_factor*iqr, q3 + iqr_factor*iqr
        iqr_flag = (x < low) | (x > hi)

        # ── 3) MAD rule ─────────────────
        med  = x.median()
        mad  = np.median(np.abs(x - med)) or 1      
        mad_z = 0.6745 * (x - med) / mad
        mad_flag = mad_z.abs() > mad_thresh

        # using OR 
        overall_out |= (z_flag | iqr_flag | mad_flag)

    return df.loc[overall_out]

