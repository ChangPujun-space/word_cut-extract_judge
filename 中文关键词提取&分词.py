#!/usr/bin/env python
# coding: utf-8

# In[148]:


import pandas as pd
import numpy as np
from jieba import analyse
import os
import jieba
from sklearn import manifold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# # 自定义词典 & 停用词词典构建

# In[238]:


##自定义携程词典
jieba.load_userdict(r"D:/Users/pjchang/Desktop/客户进线需求自动识别/stopwords-master/电话自定义词典&停用词词典/add_word.txt")

##停用词列表
g = open("D:/Users/pjchang/Desktop/客户进线需求自动识别/stopwords-master/电话自定义词典&停用词词典/cn_stopwords.txt",encoding = 'utf-8')
stopwordlist = []
for i in g.readlines():
    i = i.strip('\n')
    stopwordlist.append(i)
stopwordlist[-5:-1]


# # 切词 关键词top10抽取

# In[224]:


datasource = pd.read_excel("D:/Users/pjchang/Desktop/客户进线需求自动识别/IM端/输入/客人进线需求_im.xlsx")
datasource[['evntid','第一通客人进线内容','员工外呼酒店内容']]


# In[225]:


c = []
for i in range(len(datasource["第一通客人进线内容"].tolist())):
    c_stripblank = str(datasource["第一通客人进线内容"].tolist()[i]).replace(" ", "")
    c.append(c_stripblank)
e = datasource["员工外呼酒店内容"].tolist()
c


# In[227]:


##切词
cut_list_c = []
for i in c :
    cutwords = jieba.lcut(str(i))
    cutwords_c = [item for item in list(cutwords) if str(item) not in stopwordlist]
    cut_list_c.append(list(set(cutwords_c)))

cut_list_e = []
for i in e :
    cutwords = jieba.lcut(str(i))
    cutwords_e = [item for item in list(cutwords) if str(item) not in stopwordlist]
    cut_list_e.append(list(set(cutwords_e)))


# In[228]:


tfidf_dataset_c = []
for i in cut_list_c:
    a =(''.join(i))
    tfidf_dataset_c.append(a)

tfidf_dataset_e = []
for i in cut_list_e:
    a =(''.join(i))
    tfidf_dataset_e.append(a)


# In[234]:


##TFIDF 调用两次 改tfidf_dataset后缀c&e
tfidf = analyse.extract_tags
keywords_list = []
keywords_Wlist = []
for i in tfidf_dataset_e:
    Swords_list = []
    keywords = tfidf(str(i),topK=10,withWeight=True,allowPOS=("n","f","s","ns","vn"))
    keywords_Wlist.append(keywords)
    for j in range(len(keywords)):
        Swords_list.append(keywords[j][0])
    keywords_list.append(Swords_list)
keywordsdf = pd.DataFrame(keywords_list)
keywords_Wdf = pd.DataFrame(keywords_Wlist)

#keywordsdf.columns = ['客关1','客关2','客关3','客关4','客关5','客关6','客关7','客关8','客关9','客关10']
#keywords_Wdf.columns = ['客关1w','客关2w','客关3w','客关4w','客关5w','客关6w','客关7w','客关8w','客关9w','客关10w']

keywordsdf.columns = ['员关1','员关2','员关3','员关4','员关5','员关6','员关7','员关8','员关9','员关10']
keywords_Wdf.columns = ['员关1w','员关2w','员关3w','员关4w','员关5w','员关6w','员关7w','员关8w','员关9w','员关10w']


# In[213]:


"""
##textrank 调用两次 改tfidf_dataset后缀c&e
tr = analyse.textrank
keywords_list = []
keywords_Wlist = []
for i in tfidf_dataset_e:
    Swords_list = []
    keywords = tr(str(i),topK=10,withWeight=True,allowPOS=("n","f","s","ns","vn"))
    keywords_Wlist.append(keywords)
    for j in range(len(keywords)):
        Swords_list.append(keywords[j][0])
    keywords_list.append(Swords_list)
keywordsdf = pd.DataFrame(keywords_list)
keywords_Wdf = pd.DataFrame(keywords_Wlist)
##
keywordsdf.columns = ['客关1','客关2','客关3','客关4','客关5','客关6','客关7','客关8','客关9','客关10']
keywords_Wdf.columns = ['客关1w','客关2w','客关3w','客关4w','客关5w','客关6w','客关7w','客关8w','客关9w','客关10w']

#keywordsdf.columns = ['员关1','员关2','员关3','员关4','员关5','员关6','员关7','员关8','员关9','员关10']
#keywords_Wdf.columns = ['员关1w','员关2w','员关3w','员关4w','员关5w','员关6w','员关7w','员关8w','员关9w','员关10w']


# In[235]:


dfresult = pd.concat([dfresult,keywordsdf], axis=1)
#dfresult = pd.concat([dfresult, keywords_Wdf], axis=1)
dfresult


# # 客户&员工 共同高频率词

# In[261]:


"""
trdfresult = pd.concat([datasource, keywordsdf], axis=1)
trdfresult = pd.concat([trdfresult, keywords_Wdf], axis=1)
trdfresult
"""


# In[215]:


#trdfresult.to_csv("D:/Users/pjchang/Desktop/textrank关键词_resultwith顺序.csv",encoding="utf_8_sig")


# In[318]:


dfresult.to_csv("D:/Users/pjchang/Desktop/tfidf关键词_resultwith顺序1024withIM词库_IM.csv",encoding="utf_8_sig")


# In[319]:


dfduplicate = dfresult[['客关1','客关2','客关3','客关4','客关5','客关6','客关7','客关8','客关9','客关10','员关1','员关2','员关3','员关4','员关5','员关6','员关7','员关8','员关9','员关10']]
extract_duplicates = []
for i in range(len(dfduplicate)):
    a = dfduplicate.iloc[i][dfduplicate.iloc[i].duplicated()][dfduplicate.iloc[i][dfduplicate.iloc[i].duplicated()].notnull()]
    extract_duplicates.append(a.tolist())
keywords_dropduplicate_df = pd.DataFrame(extract_duplicates)


# In[320]:


keywords_dropduplicate_df


# In[321]:


dfwithduplicate = pd.concat([dfresult, keywords_dropduplicate_df], axis=1)
dfwithduplicate.to_csv("D:/Users/pjchang/Desktop/客户进线需求自动识别/tfidf关键词_resultwith顺序with重复词1024withIM词库_IM.csv",encoding="utf_8_sig")


# In[ ]:





# In[ ]:





# # 【Tag-word】 词典读取

# In[214]:


datasourceCI = pd.read_excel("D:/Users/pjchang/Desktop/客户进线需求自动识别/标签词典.xlsx",sheet_name="更新标签词库")
datasourceCI


# In[217]:


C_tag = set(datasourceCI['Tag'].tolist())
C_tag = list(C_tag)

C_tag_l_all = []
for j in range(len(C_tag)):
    C_tag_lt = []
    for i in range(len(datasourceCI[datasourceCI['Tag'] == C_tag[j]])):
        dfci = (datasourceCI[datasourceCI['Tag'] == C_tag[j]].iloc[i])
        C_tag_lt += (list(dfci[2:].dropna()))
    
    C_tag_l = list(set(C_tag_lt))
    
    C_tag_l_all.append(C_tag_l)


# In[220]:


DFCI = pd.DataFrame(C_tag_l_all)
DFCI.insert(0,'Tag',C_tag)


# In[69]:


#DFCI.to_csv("D:/Users/pjchang/Desktop/客户进线需求自动识别/标签词典.csv",encoding="utf_8_sig")


# # 判断逻辑模块

# In[221]:


dict_ci = {}
for i in DFCI['Tag']:
    dict_ci [str(i)] = list(DFCI[DFCI['Tag'] == str(i)].iloc[0,1:].dropna())


# In[236]:


##判断命中词组模块
def contrast(l,diction):
    result = []
    for key in diction.keys():
        lj = (set(l)&set(diction[key]))
        if len(list(lj)) != 0:
            result.append(key)
    result1 = list(set(result))
    return result1

def special_rule(tag_list):
    if "预订查询-酒店预订" in tag_list:
        #tag_list.remove("酒店政策-入住政策-取消政策")
        tag_list = ["预订查询-酒店预订"]
    return tag_list

r_list = []
for i in range(len(dfresult)):
    l1 = list(dfresult.iloc[i].dropna())
    a = contrast(l1,dict_ci)
    b = special_rule(a)
    r_list.append(b)
len(r_list)


# In[237]:


df_finalresult = pd.concat([datasource[['evntid','第一通客人进线内容','员工外呼酒店内容']], pd.DataFrame(r_list)], axis=1)
df_finalresult.to_csv("D:/Users/pjchang/Desktop/客户进线需求自动识别/IM端/IM_判断结果.csv",encoding="utf_8_sig")

