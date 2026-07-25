import nltk
import pandas as pd
import os
from PyPDF2 import PdfReader
import numpy as np

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

#------------ filter n-grams ---------------
def filter_ngram_dfs(global_map, document_count_map, union_words):
    stop_tokens = set(union_words)  
    filtered_dfs = {}

    for n in range(1, 5):
        filtered_grams = {}
        filtered_docs = {}
        for gram, count in global_map[n].items():
            tokens = gram.split()
            if not any(token in stop_tokens for token in tokens):
                filtered_grams[gram] = count
                filtered_docs[gram] = document_count_map[n].get(gram, 0)
        
        df = pd.DataFrame({
            "gram_count": pd.Series(filtered_grams),
            "document_count": pd.Series(filtered_docs)
        })
        df["Average_freq"] = df["gram_count"] / df["document_count"]
        filtered_dfs[n] = df.sort_values("gram_count", ascending=False)
    
    return filtered_dfs


def keep_zscore_outliers(df, z_thresh=1.5, cols=('gram_count', 'document_count', 'Average_freq')):
    df = df.copy()
    overall_out = pd.Series(True, index=df.index)
    for col in cols[0:2]:
        x = df[col]
        z = (x - x.mean()) / x.std(ddof=0)
        z_flag = z > z_thresh
        overall_out &= z_flag
        #df[f'zscore_{col}'] = z
    return df.loc[overall_out]

def remove_zscore_outliers(df, z_thresh=2, cols=('gram_count', 'document_count', 'Average_freq')):
    df = df.copy()
    overall_out = pd.Series(False, index=df.index)
    for col in cols:
        x = df[col]
        x = np.log(x)
        z = (x - x.mean()) / x.std(ddof=0)
        z_flag = z.abs() > z_thresh
        overall_out |= z_flag
        #df[f'zscore_{col}'] = z
    return df.loc[~overall_out]

def maha_iqr_cutoff(arr, iqr_factor=3):
    q1 = np.percentile(arr, 25, method='nearest')
    q3 = np.percentile(arr, 75, method='nearest')
    iqr = q3-q1
    cutoff = q3 + iqr_factor*iqr
    return cutoff
 
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
uni_gram_df_withmaha = uni_gram_df.copy()
uni_gram_df_withmaha['maha_dist'] = maha_uni
uni_df_outliers_maha = uni_gram_df[maha_uni > cutoff_uni] 

# z score 
uni_gram_df_outliers_zscore = keep_zscore_outliers(uni_gram_df)

from nltk.corpus import stopwords

nltk.download('stopwords')

# Filter stopwords 
uni_stopword_df = uni_gram_df[uni_gram_df.index.isin(stopwords.words('english'))]

# manually removed technical terms 
uni_gram_df_outliers_zscore_1_5_filter = pd.read_csv('uni_gram_df_outliers_zscore_1.5_filter.csv', index_col=0)

# union of our stopwords + nltk stopwords
union_words = uni_stopword_df.index.union(uni_gram_df_outliers_zscore_1_5_filter.index)

filter_ngrams = filter_ngram_dfs(global_map, document_count_map, union_words)

uni_gram_df_filtered = filter_ngrams[1]
bi_gram_df_filtered = filter_ngrams[2]
tri_gram_df_filtered = filter_ngrams[3]
quad_gram_df_filtered = filter_ngrams[4]

bi_gram_df_filtered_minimal = bi_gram_df_filtered.drop(
    bi_gram_df_filtered[bi_gram_df_filtered['gram_count'].isin([1, 2, 3])].index
)

tri_gram_df_filtered_minimal = tri_gram_df_filtered.drop(
    tri_gram_df_filtered[tri_gram_df_filtered['gram_count'].isin([1, 2, 3])].index
)

quad_gram_df_filtered_minimal = quad_gram_df_filtered.drop(
    quad_gram_df_filtered[quad_gram_df_filtered['gram_count'].isin([1, 2, 3])].index
)

uni_gram_df_filtered.to_csv('uni_gram_df_filtered.csv')
bi_gram_df_filtered.to_csv('bi_gram_df_filtered.csv')
tri_gram_df_filtered.to_csv('tri_gram_df_filtered.csv')
quad_gram_df_filtered.to_csv('quad_gram_df_filtered.csv')

bi_gram_df_filtered_minimal.to_csv('bi_gram_df_filtered_minimal.csv')
tri_gram_df_filtered_minimal.to_csv('tri_gram_df_filtered_minimal.csv')
quad_gram_df_filtered_minimal.to_csv('quad_gram_df_filtered_minimal.csv')

