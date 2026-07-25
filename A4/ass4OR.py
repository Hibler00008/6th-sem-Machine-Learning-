import fitz  
import nltk
import pandas as pd
import os
from PyPDF2 import PdfReader
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


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
file_list = os.listdir(r"D:\Books\TY\ML Lab\Assignment 4 ML\New folder")

global_map = {1:{}, 2:{}, 3:{}, 4:{}}
document_count_map = {1:{}, 2:{}, 3:{}, 4:{}}

for file in file_list:
    my_file = os.path.join(r"D:\Books\TY\ML Lab\Assignment 4 ML\New folder", file)
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
    sum = 0
    length = len(var1)
    for i in range(length):
        sum = sum + (var1[i] * var2[i])
    return sum/(length-1)

def center_the_data(var):
    varm = var - np.mean(var)
    return varm

def mahalanobis_dist(mat1, mat2, mat3, cinv):
    maha = np.zeros(len(mat1))
    for i in range(len(mat1)):
        x = np.array([mat1[i], mat2[i], mat3[i]])     
        maha[i] = x @ cinv @ x.T                       
    return np.sqrt(maha)       

# ---------------------- mahalanobis outliers for unigrams -----------------------------------

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

# chi2 threshold for p = 3 variables 
cutoff_uni = np.sqrt(chi2.ppf(0.975, df=3))            

# flag & drop outliers
uni_gram_df['maha_dist'] = maha_uni
uni_df_outliers_maha = uni_gram_df[maha_uni > cutoff_uni] 

# z score 
uni_gram_df_outliers_zscore = keep_zscore_outliers(uni_gram_df)

# iqr
uni_gram_df_outliers_iqr = keep_iqr_outliers(uni_gram_df)

# mad
uni_gram_df_outliers_mad = keep_mad_outliers(uni_gram_df)

print(covar_uni)


plt.hist(return_log_zscore(uni_gram_df),100)
plt.show()


# =============================================================================
# quantile_list = []
# 
# for i in range(100):
#     q = np.quantile(uni_gram_df['gram_count'], (i+1)/100.0)
#     quantile_list.append(q)
#     
# =============================================================================






