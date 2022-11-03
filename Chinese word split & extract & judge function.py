#!/usr/bin/env python
# coding: utf-8

# In[623]:


import pandas as pd
import numpy as np
from jieba import analyse
import os
import jieba
from sklearn import manifold
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# # 自定义词典 & 停用词词典构建

# In[624]:


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

# In[651]:


#datasource = pd.read_excel("D:/Users/pjchang/Desktop/客户进线需求自动识别/电话端/输入/客人进线需求dataset1024.xlsx")
#Cdatasource = pd.read_excel("D:/Users/pjchang/Desktop/客户进线需求自动识别/1024_IM_CALL_抽样300_tfidf_分词_员工话术.xlsx",sheet_name='电话')
Idatasource = pd.read_excel("D:/Users/pjchang/Desktop/客户进线需求自动识别/1024_IM_CALL_抽样300_tfidf_分词_员工话术.xlsx",sheet_name='IM')

datasource = Idatasource[['evntid','第一通客人进线内容','员工外呼酒店内容']]
#Idatasource[['evntid','第一通客人进线内容','员工外呼酒店内容']]


# In[652]:


c = []
for i in range(len(datasource["第一通客人进线内容"].tolist())):
    c_stripblank = str(datasource["第一通客人进线内容"].tolist()[i]).replace(" ", "")
    c.append(c_stripblank)
e = datasource["员工外呼酒店内容"].tolist()


# In[653]:


##切词
cut_list_c = []
for i in c :
    cutwords = jieba.lcut(str(i))
    cutwords_c = [item for item in list(cutwords) if str(item) not in stopwordlist]
    cut_list_c.append(list(cutwords_c))

cut_list_e = []
for i in e :
    cutwords = jieba.lcut(str(i))
    cutwords_e = [item for item in list(cutwords) if str(item) not in stopwordlist]
    cut_list_e.append(list(cutwords_e))


# In[654]:


tfidf_dataset_c = []
for i in cut_list_c:
    a =(''.join(i))
    tfidf_dataset_c.append(a)

tfidf_dataset_e = []
for i in cut_list_e:
    a =(''.join(i))
    tfidf_dataset_e.append(a)


# In[657]:


##TFIDF 调用两次 改tfidf_dataset后缀c&e
tfidf = analyse.extract_tags
keywords_list = []
keywords_Wlist = []
for i in tfidf_dataset_e:
    Swords_list = []
    keywords = tfidf(str(i),topK=10,withWeight=True,allowPOS=("n","f","s","ns","vn","an","i","ng","nz","vg"))
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


# In[658]:


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

# In[635]:


datasourceCI = pd.read_excel("D:/Users/pjchang/Desktop/客户进线需求自动识别/stopwords-master/标签词典.xlsx",sheet_name="更新标签词库")
datasourceCI


# In[636]:


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


# In[637]:


DFCI = pd.DataFrame(C_tag_l_all)
DFCI.insert(0,'Tag',C_tag)


# In[69]:


#DFCI.to_csv("D:/Users/pjchang/Desktop/客户进线需求自动识别/标签词典.csv",encoding="utf_8_sig")


# # 判断逻辑模块

# In[638]:


dict_ci = {}
for i in DFCI['Tag']:
    dict_ci [str(i)] = list(DFCI[DFCI['Tag'] == str(i)].iloc[0,1:].dropna())


# In[659]:


##判断命中词组模块
def contrast(l,diction):
    result = []
    Capture_Tag = []
    for key in diction.keys():
        lj = (set(l)&set(diction[key]))
        if len(list(lj)) != 0:
            result.append(key)
            Capture_Tag.append(list(lj))
    result1 = list(set(result))
    return result1, Capture_Tag

def special_rule(tag_list):
    if "预订查询-酒店预订" in tag_list:
        #tag_list.remove("酒店政策-入住政策-取消政策")
        tag_list = ["预订查询-酒店预订"]
    return tag_list

Capture_list = []
r_list = []
for i in range(len(dfresult)):
    l1 = list(dfresult.iloc[i].dropna())
    a = contrast(l1,dict_ci)
    b = special_rule(a[0])
    r_list.append(b)
    Capture_list.append(a[1])
print(len(r_list))
print(len(Capture_list))


# In[660]:


df_finalresult = pd.concat([datasource, pd.DataFrame(r_list), pd.DataFrame(Capture_list)], axis=1)
df_finalresult.to_csv("D:/Users/pjchang/Desktop/客户进线需求自动识别/IM_模型判断结果1102.csv",encoding="utf_8_sig")


# # N-Around

# In[546]:


Tag_all = []
for i in range(len(C_tag_l_all)):
    Tag_all = Tag_all + C_tag_l_all[i]
Tag_all


# In[547]:


N_follow_dataset_c = []
for i in range(len(cut_list_c)):
    N_follow_dataset_c.extend(cut_list_c[i])

N_follow_dataset_e = []
for i in range(len(cut_list_e)):
    N_follow_dataset_e.extend(cut_list_e[i])


# In[548]:


Tag_capture_N3_Cresult = []
for tag in Tag_all:
    Tag_capture_N3_Clist = []
    for i in [i for i,x in enumerate(N_follow_dataset_c) if x== tag]:
        try:
            Tag_capture_N3_Clist.append(N_follow_dataset_c[i-1])
        except:
            try:
                Tag_capture_N3_Clist.append(N_follow_dataset_c[i+1])
            except:
                try:
                    Tag_capture_N3_Clist.append(N_follow_dataset_c[i+2])
                except:
                    try:
                        Tag_capture_N3_Clist.append(N_follow_dataset_c[i+3])
                    except:
                        break
    for i in [i for i,x in enumerate(N_follow_dataset_e) if x== tag]:
        try:
            Tag_capture_N3_Clist.append(N_follow_dataset_e[i-1])
        except:
            try:
                Tag_capture_N3_Clist.append(N_follow_dataset_e[i+1])
            except:
                try:
                    Tag_capture_N3_Clist.append(N_follow_dataset_e[i+2])
                except:
                    try:
                        Tag_capture_N3_Clist.append(N_follow_dataset_e[i+3])
                    except:
                        break
    Tag_capture_N3_Clist.insert(0,tag)
    Tag_capture_N3_Cresult.append(Tag_capture_N3_Clist)


# In[549]:


Tag_capture_N3_Cresult[0]


# In[550]:


Tag_P_dict = []
for i in range(len(Tag_capture_N3_Cresult)):
    Tag_p_dic = {}
    for target in list(set(Tag_capture_N3_Cresult[i])):
        Tag_P = (Tag_capture_N3_Cresult[i].count(target)/(len(Tag_capture_N3_Cresult[i])))
        Tag_p_dic[str(Tag_capture_N3_Cresult[i][0] + "+" + target)] = Tag_P
    Tag_P_dict.append(sorted(Tag_p_dic.items(),key = lambda d :d[1],reverse=True))


# In[551]:


df1 = pd.DataFrame(Tag_P_dict[0])
for df_t in Tag_P_dict[1:]:
    df_t = pd.DataFrame(df_t)
    df1 = pd.concat([df1,df_t],axis=1,ignore_index=True)
df1


# In[552]:


df1.to_csv("D:/Users/pjchang/Desktop/客户进线需求自动识别/N3gram_result_CALL.csv",encoding="utf_8_sig")

