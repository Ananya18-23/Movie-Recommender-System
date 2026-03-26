#!/usr/bin/env python
# coding: utf-8

# In[155]:


# This Python 3 environment comes with many helpful analytics libraries installed
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import numpy as np   # linear algebra
import pandas as pd   # data processing, CSV file I/O (e.g. pd.read_csv)


# In[156]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[4]:


movies.head(1)


# In[5]:


credits.head(1)


# In[6]:


movies=movies.merge(credits,on='title')


# In[7]:


movies.head(1)


# In[8]:


movies.shape


# In[9]:


credits.shape


# In[10]:


movies['original_language'].value_counts()


# In[11]:


# genres
# id
# keywords
# title
# overview
# cast
# crew

movies=movies[['genres','movie_id','title','overview','keywords','cast','crew']]


# In[12]:


movies.head()


# In[13]:


movies.info()


# In[14]:


movies.isnull().sum()


# In[15]:


movies.dropna(inplace=True)


# In[16]:


movies.duplicated().sum()


# In[17]:


movies['genres']


# In[18]:


# [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"},{"id": 28, "name": "Action"},{"id": 28, "name": "Action"}]
# ['action','adventure','ffantasy','scify']


# In[19]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"},{"id": 12, "name": "Adventure"},{"id": 28, "name": "Action"},{"id": 28, "name": "Action"}]')


# In[20]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L    
    


# In[21]:


movies['genres']=movies['genres'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['keywords']=movies['keywords'].apply(convert)


# In[24]:


movies.head()


# In[25]:


def convert3(obj):
    L=[]
    counter=0
    for  i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    
    return L   


# In[26]:


movies['cast']=movies['cast'].apply(convert3)


# In[27]:


movies.head()


# In[28]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            
           L.append(i['name'])
           break
    return L  


# In[29]:


movies['crew']=movies['crew'].apply(fetch_director)


# In[30]:


movies.head()


# In[31]:


movies['overview'][0]


# In[32]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[33]:


movies.head()


# In[34]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[35]:


movies.head()


# In[40]:


movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[41]:


movies.head()


# In[42]:


new_df=movies[['movie_id','title','tags']]


# In[43]:


new_df


# In[45]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[46]:


new_df['tags']


# In[47]:


new_df.head()


# In[75]:


pip install nltk


# In[76]:


import nltk


# In[77]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[102]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[103]:


new_df['tags'][0]


# In[104]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[105]:


new_df.head()


# In[106]:


new_df['tags'][0]


# In[107]:


new_df['tags'][1]


# In[120]:


from sklearn.feature_extraction.text import CountVectorizer


# In[121]:


cv=CountVectorizer(max_features=5000,stop_words='english')


# In[122]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[123]:


vectors


# In[124]:


vectors[0]


# In[130]:


cv.get_feature_names_out()


# In[126]:


['loved','loving','love']
['love','love','love']


# In[127]:


ps.stem('danced')


# In[128]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[131]:


from sklearn.metrics.pairwise import cosine_similarity


# In[135]:


similarity=cosine_similarity(vectors)


# In[146]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[151]:


def recommand(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        print(new_df.iloc[i[0]].title)
    return


# In[152]:


recommand('Avatar')


# In[154]:


recommand('Batman Begins')


# In[153]:


new_df.iloc[1216].title


# In[158]:


import pickle


# In[165]:


pickle.dump(new_df.to_dict(),open('movies_dict.pkl','wb'))


# In[166]:


new_df['title'].values


# In[167]:


new_df.to_dict()


# In[168]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




