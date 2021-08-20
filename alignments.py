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

import umap.umap_ as umap

import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import altair as alt  
import bokeh
import plotly.express as px
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.figure_factory as ff
import kmeans1d
import seaborn as sns
import re
seq_file = st.file_uploader("Upload FASTA File",type = [".fastq"])


clone = st.text_input("Clone sequence ",\
    "GCCTCTGGATTCACCTTTAGTAACTATTGCATGAGCTGGGTCCGCCAGGCTCCAGGGAAGGGGCTGGAGTGGG\n\
    TGGCCAACATAAAGCAAGATGGATGTGAGAAATACTATGTGGACTCTGTGAAGGGCCGATTCCCCATCTCCAGAGA\n\
    CAACGCCAAGAACTCACTGTATCTGCAAATGAACAGCCTGAGAGCCGAGGACACGGCTGTGTATTACTGTGCGAGAG\n\
    ATCTGAGGAACGAAATGGCGGGGGATGGTGCGGGGAGTTAGTCGACTACTACTACTACTACATGTACGTCTGGGGCAAAGGGACCAC")

clone=re.sub('[ ]',"",clone)
clone=re.sub('\n','',clone) 
st.write('input clone sequence is: '+clone)
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


# initializing K 
K = st.slider('Kmer length',10,len(clone),np.int(np.round(len(clone)-150)))

  

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
    st.write(str(ses.shape[0])+' sequences in fastq')
    ses.rename(columns={0:'seq'},inplace=True)
    
    ses=ses.groupby(['seq'],as_index=False).size()
    filtdepth=st.selectbox('filter sequences below depth of',np.arange(0,10),4)
    ses=ses[ses['size']>=filtdepth]
    ses.reset_index(inplace=True)      
    ses.rename(columns={0:'seq'},inplace=True)
    st.write(str(ses['size'].sum())+' sequences in fastq with >='+str(filtdepth)+' read and '+str(ses.shape[0])+' unique sequences')
    p_bar=st.progress(0)
    ppp=dict()
    ppp2=dict()
    ppp3=dict()
    lr=len(res)
    p_progress=0
    st.write('Exact matching progress wth kmer length '+str(K))

    retrievednum=0
    cols=st.columns(4)
    usecol=np.tile(np.arange(0,4),200)
    
    indusecol=0
    fullv=True
    if fullv:
        for jj in res:
            p_bar.progress((1+p_progress)/lr)
            ppp[p_progress]=pd.DataFrame(ses[ses['seq'].str.contains(jj)].index)
            ppp2[p_progress]=pd.DataFrame(ses['seq'].str.contains(jj))
            ppp3[p_progress]=pd.DataFrame(ses[ses['seq'].str.contains(jj)])
            retrievednum=pd.concat(ppp,axis=0).drop_duplicates().shape[0]
            if (p_progress-1)%10 == 0:
                cols[usecol[indusecol]].write('with '+str(p_progress+1)+' kmers # sequences retrieved '+str(retrievednum))
                indusecol+=1
            p_progress+=1
    else:
        myf=open('myfile.pkl','rb')   
        ppp2,ppp=pickle.load(myf)   
   
    prematch_vector=pd.concat(ppp2,axis=1)
    keepin=prematch_vector.apply(lambda x: any(x),axis=1)
    match_vector=prematch_vector[keepin]
    #seqkeep=ses['seq'][match_vector.index]
    seqkeep=ses.iloc[match_vector.index]
    #hist_data = [len(str(jjj)) for jjj in ses['seq'][match_vector.index].apply(lambda x: str(x))]
    hist_data = [len(str(jjj)) for jjj in np.repeat(seqkeep['seq'],seqkeep['size'])]
    fig, ax = plt.subplots()
    ax.hist(hist_data, bins=50)
    st.pyplot(fig)
    #fig = ff.create_distplot([hist_data], ['distribution of retrieved lengths'], bin_size=1)
   
    #st.plotly_chart(fig)
    ncuse = st.selectbox('Number of clusters based on length', [1,2,3],2)
    clusters, centroids = kmeans1d.cluster(hist_data,ncuse)
    clusterer=PCA(n_components=2)
    alntrans=clusterer.fit_transform(match_vector)
    #pced=pd.DataFrame(alntrans,columns=['0','1'])
    pced=pd.DataFrame({'seq': np.repeat(seqkeep['seq'],seqkeep['size'])})
   
    pced['length']=hist_data
    ##preds=clusterer.fit_predict(alntrans)
    pced['prediction']=clusters
    pced.drop_duplicates(inplace=True)
    pced['depth']=seqkeep['size']
    #grouped = pced.groupby(pced['prediction'])
    
    ldist=list()
    difgrou=list()
    seqgrou=list()
    for iii in np.unique(pced['prediction']):
        #individualcluster=grouped.get_group(iii)['length']
        
        #ldist.append(individualcluster[::20])
        ldist.append(pced['length'][pced['prediction']==iii])
        difgrou.append(match_vector.iloc[np.where(pced['prediction']==iii)])
        seqgrou.append(seqkeep.iloc[np.where(pced['prediction']==iii)])
        
    fig = ff.create_distplot(ldist,group_labels=np.unique(pced['prediction']).tolist())
    st.write('Clusters')
    st.plotly_chart(fig)
    fit=umap.UMAP()  
    jjj=0
    
    for iii in np.unique(pced['prediction']):
        
        st.write('Group '+str(iii)+' dimension reduction:')
        gg=difgrou[jjj].groupby(difgrou[jjj].columns.tolist(),as_index=False).size() 
        
        #gseq=pd.DataFrame(seqgrou[jjj]).groupby(pd.DataFrame(seqgrou[jjj]).columns.tolist(),as_index=False).size()  
        gseq=seqgrou[jjj]
        hm=sns.clustermap(gg.drop('size',axis=1).T,row_cluster=False) 
        st.pyplot(hm)
        ff=fit.fit_transform(gg.drop('size',axis=1))  
        ff=pd.DataFrame(ff)
        pjj=px.scatter(x=ff[0],y=ff[1],size=np.sqrt(gg['size']),color=gg['size'])
        st.plotly_chart(pjj)
        gseq=gseq.sort_values(['size'],ascending=False)
        mmm={}
        slq=[]
        countercc=0
        p_bar=st.progress(0)
        for predseq in gseq['seq']:
            if countercc % 100 == 0:
                p_bar.progress(countercc/gseq.shape[0])
                st.write(countercc)

            mmm[predseq]=pairwise2.align.localxx(predseq,clone)[0]
            bb=format_alignment(*mmm[predseq],full_sequences=True).split('\n')  
            slq.append(bb[3])
            countercc+=1
        with st.expander(str(iii)+' alignment group'):
           
            st.table(pd.DataFrame(slq,gseq['size']))
        jjj+=1

