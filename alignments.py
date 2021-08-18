import streamlit as st
import neatbio.sequtils as utils
from Bio.Seq import Seq
from Bio import SeqIO
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
from collections import Counter
import base64, zlib
# data pkgs
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use("Agg")
import numpy as np
import pdb
import gzip
import shutil
from itertools import islice
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import altair as alt  
import bokeh
import plotly.express as px
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

seq_file = st.file_uploader("Upload FASTA File",type = [".fastq"])


clone = st.text_input("Clone sequence ","CAGCCTCTGGTTTCACCTTTAGTAACTATTGCATGAGCTGGGTCCGCCCGGCTCCAGGGAAGGGGCTGGAGTGGGTGGCCCACATAAAGCAAGATGG")


# Python3 code to demonstrate working of
# Overlapping consecutive K splits
# Using islice() + generator function + join() 

  
# generator function 
def over_slice(test_str, K):
    itr = iter(test_str)
    res = tuple(islice(itr, K))
    if len(res) == K:
        yield res    
    for ele in itr:
        res = res[1:] + (ele,)
        yield res
  

  
# printing original string
st.write("Clone: " + str(clone))
  
# initializing K 
K = st.slider('Kmer length',10,100,48)

  

res = ["".join(ele) for ele in over_slice(clone, K)]
pd_res=pd.DataFrame(res)  
pd_res.rename(columns={0:'Seq_kmer'},inplace=True)

# printing result
st.write("Overlapping windows:")
with st.expander('table of kmer with length '+str(K)):
    st.table(pd_res)



if seq_file is not None:
    lll=list()
    for line in seq_file:
        lll.append(line.decode("utf-8"))

    l2=pd.DataFrame(lll)
    l2.reset_index(inplace=True)
    ses=pd.DataFrame(l2.iloc[np.arange(1,l2.shape[0],4)]) 
    ses.reset_index(inplace=True)      
    ses.rename(columns={0:'seq'},inplace=True)
    st.write(ses.iloc[0:20])
    p_bar=st.progress(0)
    ppp=dict()
    ppp2=dict()
    lr=len(res)
    p_progress=0
    st.write('Exact matching progress wth kmer length '+str(K))
    for jj in res:
        p_bar.progress((1+p_progress)/lr)
        
        ppp[p_progress]=pd.DataFrame(ses[ses['seq'].str.contains(jj)].index)
        ppp2[p_progress]=pd.DataFrame(ses['seq'].str.contains(jj))
        p_progress+=1
           
    
    prematch_vector=pd.concat(ppp2,axis=1)
    match_vector=prematch_vector[prematch_vector.apply(lambda x: any(x),axis=1)]
    
    fit = umap.UMAP()
    
    alntrans=fit.fit_transform(match_vector)
    list_score=[]
    range_n_clusters = list (range(2,12))
    for n_clusters in range_n_clusters:
        clusterer = KMeans(n_clusters=n_clusters)
        preds = clusterer.fit_predict(alntrans)       #fitting the algorithm to the scaled values rather than the actual values
        centers = clusterer.cluster_centers_

        score = silhouette_score(alntrans, preds)
        list_score.append(score)      
        print("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))
    umapped=pd.DataFrame(alntrans,columns=['0','1'])
    ncuse=np.argmax(list_score)+2
    clusterer=KMeans(n_clusters=ncuse)
    preds=clusterer.fit_predict(alntrans)
    umapped['prediction']=preds

    umapped['seq']=[str(jjj) for jjj in ses['seq'][match_vector.index].apply(lambda x: str(x))]
    cola,colb=st.columns(2)
    st.write(str(ncuse)+' clusters with values:')
    st.write(pd.DataFrame(umapped.prediction.value_counts()))
    st.write(px.scatter(umapped,x='0',y='1',color='prediction',hover_data=['seq'])) 
    hm=sns.clustermap(match_vector) 
    st.pyplot(hm)

    lll=dict()
    for jj in np.unique(preds):
        mmm=dict()
        lll[jj]=list()
        for predseq in umapped['seq'][umapped['prediction']==jj]:
            
            mmm[predseq]=pairwise2.align.localxx(predseq,clone)[0]
            bb=format_alignment(*mmm[predseq],full_sequences=True).split('\n')  
            lll[jj].append(bb[3])
        with st.expander(str(jj)+' alignment group'):
            st.table(lll[jj])
        
