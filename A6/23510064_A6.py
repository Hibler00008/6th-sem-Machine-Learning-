import fitz  
import nltk
import pandas as pd
import os
from PyPDF2 import PdfReader
import numpy as np
import matplotlib.pyplot as plt

def ngrams_extract(myText):
    tokens = [token.lower() for token in nltk.word_tokenize(myText) if token.isalnum()]
    unigrams_map = {}
    bigrams_map = {}
    trigrams_map = {}
    quadgrams_map = {}
    for i in range(len(tokens)):
        if i < len(tokens):
            unigram = tokens[i]
            unigrams_map[unigram] = unigrams_map.get(unigram, 0) + 1
        if i < len(tokens) - 1:
            bigram = f"{tokens[i]} {tokens[i+1]}"
            bigrams_map[bigram] = bigrams_map.get(bigram, 0) + 1
        if i < len(tokens) - 2:
            trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            trigrams_map[trigram] = trigrams_map.get(trigram, 0) + 1
        if i < len(tokens) - 3:
            quadgram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]} {tokens[i+3]}"
            quadgrams_map[quadgram] = quadgrams_map.get(quadgram, 0) + 1
    return {1:unigrams_map, 2:bigrams_map, 3:trigrams_map, 4:quadgrams_map}

# def extract_pdf_text(pdf_file):
#     reader = fitz.open(pdf_file)
#     pdf_text = ""
#     for page in reader:
#         pdf_text += page.get_text() + "\n"
#     reader.close()
#     return pdf_text

def extract_file_ngrams(my_file):

    pdf_file = open(my_file, 'rb')
    reader = PdfReader(pdf_file)
    
    text = ''
    for page in reader.pages: 
        text += page.extract_text().lower()
    
    gram_map = ngrams_extract(text)
    return gram_map

file_gram_map = {}
file_list = os.listdir(r"D:\Books\TY\ML Lab\Assignment 6 ML\merged_all")

global_map = {1:{}, 2:{}, 3:{}, 4:{}}
document_count_map = {1:{}, 2:{}, 3:{}, 4:{}}

for file in file_list:
    my_file = os.path.join(r"D:\Books\TY\ML Lab\Assignment 6 ML\merged_all", file)
    file_gram_map[my_file] = extract_file_ngrams(my_file)

    for i in range(1,5):
        i_gram = file_gram_map[my_file][i]
        for key in i_gram:
            count = i_gram[key]
            if key in global_map[i]:
                global_map[i][key] += count
            else:
                global_map[i][key] = count
                
            document_count_map[i][key] = document_count_map[i].get(key, 0) + 1

                
uni_document_count_series = pd.Series(document_count_map[1])
uni_gram_count_series = pd.Series(global_map[1])
uni_gram_df = pd.DataFrame({"gram_count":uni_gram_count_series,"document_count":uni_document_count_series})
uni_gram_df["Average_freq"] = uni_gram_df["gram_count"]/uni_gram_df["document_count"]

bi_document_count_series = pd.Series(document_count_map[2])
bi_gram_count_series = pd.Series(global_map[2])
bi_gram_df = pd.DataFrame({"gram_count":bi_gram_count_series,"document_count":bi_document_count_series})
bi_gram_df["Average_freq"] = bi_gram_df["gram_count"]/bi_gram_df["document_count"]

tri_document_count_series = pd.Series(document_count_map[3])
tri_gram_count_series = pd.Series(global_map[3])
tri_gram_df = pd.DataFrame({"gram_count":tri_gram_count_series,"document_count":tri_document_count_series})
tri_gram_df["Average_freq"] = tri_gram_df["gram_count"]/tri_gram_df["document_count"]

quad_document_count_series = pd.Series(document_count_map[4])
quad_gram_count_series = pd.Series(global_map[4])
quad_gram_df = pd.DataFrame({"gram_count":quad_gram_count_series,"document_count":quad_document_count_series})
quad_gram_df["Average_freq"] = quad_gram_df["gram_count"]/quad_gram_df["document_count"]

def keep_zscore_outliers(df, z_thresh=2, cols=('gram_count', 'document_count', 'Average_freq')):
    df = df.copy()
    overall_out = pd.Series(False, index=df.index)
    for col in cols:
        x = df[col]
        x = np.log(x)
        z = (x - x.mean()) / x.std(ddof=0)
        z_flag = z.abs() > z_thresh
        overall_out |= z_flag
        df[f'zscore_{col}'] = z
    return df.loc[overall_out]

def return_log_zscore(df, z_thresh=2, cols=('gram_count', 'document_count', 'Average_freq')):
    df = df.copy()
    x = df['gram_count']
    x = np.log(x)
    return x
    

def keep_iqr_outliers(df, iqr_factor=3, cols=('gram_count', 'document_count', 'Average_freq')):
    df = df.copy()
    overall_out = pd.Series(False, index=df.index)
    for col in cols:
        x = df[col]
        q1, q3 = x.quantile([0.25, 0.75])
        iqr     = q3 - q1
        low, hi = q1 - iqr_factor*iqr, q3 + iqr_factor*iqr
        iqr_flag = (x < low) | (x > hi)
        overall_out |= iqr_flag
    return df.loc[overall_out]

def maha_iqr_cutoff(arr, iqr_factor=3):
    q1 = np.percentile(arr, 25, method='nearest')
    q3 = np.percentile(arr, 75, method='nearest')
    iqr = q3-q1
    cutoff = q3 + iqr_factor*iqr
    return cutoff

def keep_mad_outliers(df,  mad_thresh=3.5, cols=('gram_count', 'document_count', 'Average_freq')):
    df = df.copy()
    overall_out = pd.Series(False, index=df.index)
    for col in cols:
        x = df[col]
        med  = x.median()
        mad  = np.median(np.abs(x - med)) or 1      
        mad_z = 0.6745 * (x - med) / mad
        mad_flag = mad_z.abs() > mad_thresh
        overall_out |= mad_flag
        df[f'mad_z_{col}'] = mad_z 
    return df.loc[overall_out]
 
def fncov(var1, var2):
    mean1 = sum(var1)/len(var1)
    mean2 = sum(var2)/len(var2)
    sum_cov = 0
    length = len(var1)
    for i in range(length):
        sum_cov += (var1[i] - mean1) * (var2[i] - mean2)
    return sum_cov / (length - 1)

def center_the_data(var):
    varm = var - np.mean(var)
    return varm

def mahalanobis_dist(mat1, mat2, mat3, cinv):
    maha = np.zeros(len(mat1))
    for i in range(len(mat1)):
        x = np.array([mat1[i], mat2[i], mat3[i]])     
        intermediate = np.matmul(x.T, cinv) 
        maha[i] = np.matmul(intermediate, x)                     
    return np.sqrt(maha)    

# ----------------------for unigrams -----------------------------------

X_uni = uni_gram_df[['gram_count','document_count','Average_freq']].values
X_mean_uni = X_uni.mean(axis=0)
X_cent_uni = X_uni - X_mean_uni   

#  compute inverse covariance matrix 
covar_uni = [[0] * 3 for i in range(3)]
lselect_uni = [uni_gram_df['gram_count'].tolist(), 
               uni_gram_df['document_count'].tolist(), 
               uni_gram_df['Average_freq'].tolist()]

for i in range(3):
    for j in range(3):
        covar_uni[i][j] = fncov(lselect_uni[i], lselect_uni[j])
                
covar_uni = np.array(covar_uni)
covar_inv_uni = np.linalg.pinv(covar_uni)

maha_uni = mahalanobis_dist(X_cent_uni[:, 0], X_cent_uni[:, 1], X_cent_uni[:, 2], covar_inv_uni)

# iqr cutoff for maha distances > q3 + 1.5*iqr 
cutoff_uni =  maha_iqr_cutoff(maha_uni)         

# flag & drop outliers
uni_gram_df['maha_dist'] = maha_uni
uni_df_outliers_maha = uni_gram_df[maha_uni > cutoff_uni] 

# z score 
uni_gram_df_outliers_zscore = keep_zscore_outliers(uni_gram_df)

# iqr
uni_gram_df_outliers_iqr = keep_iqr_outliers(uni_gram_df)

# mad
uni_gram_df_outliers_mad = keep_mad_outliers(uni_gram_df)

# print(covar_uni)

# plt.hist(return_log_zscore(uni_gram_df),100)
# plt.show()

# ---------------- Zscore comparison ----------------
# 1) Words present in maha results but not in zscore -> false positives
uni_false_positive_zscore = uni_df_outliers_maha.loc[
    uni_df_outliers_maha.index.difference(uni_gram_df_outliers_zscore.index)
].copy()
uni_false_positive_zscore['tag'] = 'false_positive'  

# 2) Words present in zscore results but not in maha -> false negatives
uni_false_negative_zscore = uni_gram_df_outliers_zscore.loc[
    uni_gram_df_outliers_zscore.index.difference(uni_df_outliers_maha.index)
].copy()
uni_false_negative_zscore['tag'] = 'false_negative'

# uni_df_outliers_maha.to_csv('uni_df_outliers_maha.csv')
# uni_gram_df_outliers_iqr.to_csv('uni_gram_df_outliers_iqr.csv')
# uni_gram_df_outliers_mad.to_csv('uni_gram_df_outliers_mad.csv')
# uni_gram_df_outliers_zscore.to_csv('uni_gram_df_outliers_zscore.csv')

# ---------------- IQR comparison ----------------
# 1) Words present in maha results but not in iqr -> false positives
uni_false_positive_iqr = uni_df_outliers_maha.loc[
    uni_df_outliers_maha.index.difference(uni_gram_df_outliers_iqr.index)
].copy()
uni_false_positive_iqr['tag'] = 'false_positive'

# 2) Words present in iqr results but not in maha -> false negatives
uni_false_negative_iqr = uni_gram_df_outliers_iqr.loc[
    uni_gram_df_outliers_iqr.index.difference(uni_df_outliers_maha.index)
].copy()
uni_false_negative_iqr['tag'] = 'false_negative'

# ---------------- MAD comparison ----------------
# 1) Words present in maha results but not in mad -> false positives
uni_false_positive_mad = uni_df_outliers_maha.loc[
    uni_df_outliers_maha.index.difference(uni_gram_df_outliers_mad.index)
].copy()
uni_false_positive_mad['tag'] = 'false_positive'

# 2) Words present in mad results but not in maha -> false negatives
uni_false_negative_mad = uni_gram_df_outliers_mad.loc[
    uni_gram_df_outliers_mad.index.difference(uni_df_outliers_maha.index)
].copy()
uni_false_negative_mad['tag'] = 'false_negative'

# ----------------------for bigrams -----------------------------------

X_bi = bi_gram_df[['gram_count','document_count','Average_freq']].values
X_mean_bi = X_bi.mean(axis=0)
X_cent_bi = X_bi - X_mean_bi   

#  compute inverse covariance matrix 
covar_bi = [[0] * 3 for i in range(3)]
lselect_bi = [bi_gram_df['gram_count'].tolist(), 
              bi_gram_df['document_count'].tolist(), 
              bi_gram_df['Average_freq'].tolist()]

for i in range(3):
    for j in range(3):
        covar_bi[i][j] = fncov(lselect_bi[i], lselect_bi[j])
                
covar_bi = np.array(covar_bi)
covar_inv_bi = np.linalg.pinv(covar_bi)

maha_bi = mahalanobis_dist(X_cent_bi[:, 0], X_cent_bi[:, 1], X_cent_bi[:, 2], covar_inv_bi)

# iqr cutoff for maha distances > q3 + 1.5*iqr 
cutoff_bi = maha_iqr_cutoff(maha_bi)           

# flag & drop outliers
bi_gram_df['maha_dist'] = maha_bi
bi_df_outliers_maha = bi_gram_df[maha_bi > cutoff_bi] 

# z score 
bi_gram_df_outliers_zscore = keep_zscore_outliers(bi_gram_df)

# iqr
bi_gram_df_outliers_iqr = keep_iqr_outliers(bi_gram_df)

# mad
bi_gram_df_outliers_mad = keep_mad_outliers(bi_gram_df)

# ---------------- Zscore comparison ----------------
bi_false_positive_zscore = bi_df_outliers_maha.loc[
    bi_df_outliers_maha.index.difference(bi_gram_df_outliers_zscore.index)
].copy()
bi_false_positive_zscore['tag'] = 'false_positive'  

bi_false_negative_zscore = bi_gram_df_outliers_zscore.loc[
    bi_gram_df_outliers_zscore.index.difference(bi_df_outliers_maha.index)
].copy()
bi_false_negative_zscore['tag'] = 'false_negative'

# ---------------- IQR comparison ----------------
bi_false_positive_iqr = bi_df_outliers_maha.loc[
    bi_df_outliers_maha.index.difference(bi_gram_df_outliers_iqr.index)
].copy()
bi_false_positive_iqr['tag'] = 'false_positive'

bi_false_negative_iqr = bi_gram_df_outliers_iqr.loc[
    bi_gram_df_outliers_iqr.index.difference(bi_df_outliers_maha.index)
].copy()
bi_false_negative_iqr['tag'] = 'false_negative'

# ---------------- MAD comparison ----------------
bi_false_positive_mad = bi_df_outliers_maha.loc[
    bi_df_outliers_maha.index.difference(bi_gram_df_outliers_mad.index)
].copy()
bi_false_positive_mad['tag'] = 'false_positive'

bi_false_negative_mad = bi_gram_df_outliers_mad.loc[
    bi_gram_df_outliers_mad.index.difference(bi_df_outliers_maha.index)
].copy()
bi_false_negative_mad['tag'] = 'false_negative'

# ----------------------for trigrams -----------------------------------

X_tri = tri_gram_df[['gram_count','document_count','Average_freq']].values
X_mean_tri = X_tri.mean(axis=0)
X_cent_tri = X_tri - X_mean_tri   

#  compute inverse covariance matrix 
covar_tri = [[0] * 3 for i in range(3)]
lselect_tri = [tri_gram_df['gram_count'].tolist(), 
               tri_gram_df['document_count'].tolist(), 
               tri_gram_df['Average_freq'].tolist()]

for i in range(3):
    for j in range(3):
        covar_tri[i][j] = fncov(lselect_tri[i], lselect_tri[j])
                
covar_tri = np.array(covar_tri)
covar_inv_tri = np.linalg.pinv(covar_tri)

maha_tri = mahalanobis_dist(X_cent_tri[:, 0], X_cent_tri[:, 1], X_cent_tri[:, 2], covar_inv_tri)

# iqr cutoff for maha distances > q3 + 1.5*iqr 
cutoff_tri = maha_iqr_cutoff(maha_tri)            

# flag & drop outliers
tri_gram_df['maha_dist'] = maha_tri
tri_df_outliers_maha = tri_gram_df[maha_tri > cutoff_tri] 

# z score 
tri_gram_df_outliers_zscore = keep_zscore_outliers(tri_gram_df)

# iqr
tri_gram_df_outliers_iqr = keep_iqr_outliers(tri_gram_df)

# mad
tri_gram_df_outliers_mad = keep_mad_outliers(tri_gram_df)

# ---------------- Zscore comparison ----------------
tri_false_positive_zscore = tri_df_outliers_maha.loc[
    tri_df_outliers_maha.index.difference(tri_gram_df_outliers_zscore.index)
].copy()
tri_false_positive_zscore['tag'] = 'false_positive'  

tri_false_negative_zscore = tri_gram_df_outliers_zscore.loc[
    tri_gram_df_outliers_zscore.index.difference(tri_df_outliers_maha.index)
].copy()
tri_false_negative_zscore['tag'] = 'false_negative'

# ---------------- IQR comparison ----------------
tri_false_positive_iqr = tri_df_outliers_maha.loc[
    tri_df_outliers_maha.index.difference(tri_gram_df_outliers_iqr.index)
].copy()
tri_false_positive_iqr['tag'] = 'false_positive'

tri_false_negative_iqr = tri_gram_df_outliers_iqr.loc[
    tri_gram_df_outliers_iqr.index.difference(tri_df_outliers_maha.index)
].copy()
tri_false_negative_iqr['tag'] = 'false_negative'

# ---------------- MAD comparison ----------------
tri_false_positive_mad = tri_df_outliers_maha.loc[
    tri_df_outliers_maha.index.difference(tri_gram_df_outliers_mad.index)
].copy()
tri_false_positive_mad['tag'] = 'false_positive'

tri_false_negative_mad = tri_gram_df_outliers_mad.loc[
    tri_gram_df_outliers_mad.index.difference(tri_df_outliers_maha.index)
].copy()
tri_false_negative_mad['tag'] = 'false_negative'

# ----------------------for quadgrams -----------------------------------

X_quad = quad_gram_df[['gram_count','document_count','Average_freq']].values
X_mean_quad = X_quad.mean(axis=0)
X_cent_quad = X_quad - X_mean_quad   

#  compute inverse covariance matrix 
covar_quad = [[0] * 3 for i in range(3)]
lselect_quad = [quad_gram_df['gram_count'].tolist(), 
                quad_gram_df['document_count'].tolist(), 
                quad_gram_df['Average_freq'].tolist()]

for i in range(3):
    for j in range(3):
        covar_quad[i][j] = fncov(lselect_quad[i], lselect_quad[j])
                
covar_quad = np.array(covar_quad)
covar_inv_quad = np.linalg.pinv(covar_quad)

maha_quad = mahalanobis_dist(X_cent_quad[:, 0], X_cent_quad[:, 1], X_cent_quad[:, 2], covar_inv_quad)

# iqr cutoff for maha distances > q3 + 1.5*iqr 
cutoff_quad = maha_iqr_cutoff(maha_quad) 

# flag & drop outliers
quad_gram_df['maha_dist'] = maha_quad
quad_df_outliers_maha = quad_gram_df[maha_quad > cutoff_quad] 

# z score 
quad_gram_df_outliers_zscore = keep_zscore_outliers(quad_gram_df)

# iqr
quad_gram_df_outliers_iqr = keep_iqr_outliers(quad_gram_df)

# mad
quad_gram_df_outliers_mad = keep_mad_outliers(quad_gram_df)

# ---------------- Zscore comparison ----------------
quad_false_positive_zscore = quad_df_outliers_maha.loc[
    quad_df_outliers_maha.index.difference(quad_gram_df_outliers_zscore.index)
].copy()
quad_false_positive_zscore['tag'] = 'false_positive'  

quad_false_negative_zscore = quad_gram_df_outliers_zscore.loc[
    quad_gram_df_outliers_zscore.index.difference(quad_df_outliers_maha.index)
].copy()
quad_false_negative_zscore['tag'] = 'false_negative'

# ---------------- IQR comparison ----------------
quad_false_positive_iqr = quad_df_outliers_maha.loc[
    quad_df_outliers_maha.index.difference(quad_gram_df_outliers_iqr.index)
].copy()
quad_false_positive_iqr['tag'] = 'false_positive'

quad_false_negative_iqr = quad_gram_df_outliers_iqr.loc[
    quad_gram_df_outliers_iqr.index.difference(quad_df_outliers_maha.index)
].copy()
quad_false_negative_iqr['tag'] = 'false_negative'

# ---------------- MAD comparison ----------------
quad_false_positive_mad = quad_df_outliers_maha.loc[
    quad_df_outliers_maha.index.difference(quad_gram_df_outliers_mad.index)
].copy()
quad_false_positive_mad['tag'] = 'false_positive'

quad_false_negative_mad = quad_gram_df_outliers_mad.loc[
    quad_gram_df_outliers_mad.index.difference(quad_df_outliers_maha.index)
].copy()
quad_false_negative_mad['tag'] = 'false_negative'

