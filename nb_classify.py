import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import codecs
import math
from operator import itemgetter
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import MultipleLocator

def get_filenamelist():
    filenamelist=[]
    labellist=[]
    for spamlabel in range(2):  # emaillabel=0,1
        if (spamlabel):
            s = 'spam'
        else:
            s = 'ham'
        for i in range (55):
            filepath = 'data/'+ s + "dir/" + s + str(i + 1) + ".txt"
            if os.path.exists(filepath):
                filenamelist.append(filepath)
                labellist.append(spamlabel)
            else:
                break
    return filenamelist,labellist

def get_one_doc_tokens(filename):
    input_file = codecs.open(filename, "r", encoding='utf-8')
    allline = ""
    for line in input_file:
        allline = allline + line.strip()
    str=allline.strip()
    doc_tokens = re.findall(r"<a.*?/a>|<[^\>]*>|[\w'@#]+", str.lower())
    tokens=[]
    for token in doc_tokens:
        temp=re.findall(r'[0-9]', token)
        if len(temp)==0 and len(token)>2: #过滤掉数字和长度小于3的单词
            tokens.append(token)
    return tokens

def write_tokens(outfile):
    if os.path.exists(outfile):
        return outfile

    filenamelist, labellist = get_filenamelist()
    with open(outfile, 'w', encoding='utf-8') as wf:
        for index in range(len(labellist)):
            wf.write(str(labellist[index]))
            tokens = get_one_doc_tokens(filenamelist[index])
            for token in tokens:
                wf.write(' {}'.format(token))
            wf.write('\n')
    wf.close()
    return outfile

tokensfile = write_tokens('data/label_tokens.txt')

def read_tokens(filename):
    labels=[]
    tokens_list_of_list=[] # list of list,每个list存储一个doc出现的tokens
    tokens_list=[] #所有doc的tokens总和
    tokens_count_doc_nums = {} #出现token的doc数量
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line_information=line.strip().split(' ')
            labels.append(line_information[0])
            tokens=line_information[1:]
            tokens_list_of_list.append(tokens)
            tokens_list += tokens
            for token in set(tokens):
                if tokens_count_doc_nums.__contains__(token):
                    tokens_count_doc_nums[token]+=1
                else:
                    tokens_count_doc_nums[token] = 1
    return labels,tokens_list_of_list,tokens_list,tokens_count_doc_nums

labels, tokens_list_of_list, tokens_list, tokens_count_doc_nums = read_tokens(tokensfile)


def get_stopwords():
    stopword_file = codecs.open('data/stopwords.txt', "r", encoding='utf-8')
    stopwords = set([line.strip() for line in stopword_file])
    return stopwords

def get_tfidf(num_docs,tokens_list,tokens_count_doc_nums):
    words_tfidf={}
    words_idf = {}
    stopwords=get_stopwords()
    for word in set(tokens_list):
        tf=float(tokens_list.count(word)) / len(tokens_list)
        idf=math.log(float(num_docs) /(1 + tokens_count_doc_nums[word]))
        tfidf=tf*idf
        if word in stopwords:
            tfidf=0
            idf=0
        words_idf[word]=idf
        words_tfidf[word]=tfidf
    words_tfidf=sorted(words_tfidf.items(), key=itemgetter(1), reverse=True)
    print('words_tfidf: ',words_tfidf)
    return words_tfidf,words_idf

words_tfidf,words_idf=get_tfidf(len(labels), tokens_list, tokens_count_doc_nums)



def generate_fecturefile(words_tfidf,tokens_list_of_list,feature_d,vecfile):
    name=['label']
    for index in range(feature_d):
        name.append(words_tfidf[index][0])

    words_vec=[]
    for index in range(len(labels)):
        list = [labels[index]]
        for word in name[1:]:
            if word in tokens_list_of_list[index]:
                list.append(1)  # 1代表word在
            else:
                list.append(0) # 0代表word不在
        words_vec.append(list)

    vecdata = pd.DataFrame(columns=name, data=words_vec)
    vecdata.to_csv(vecfile, index=False)
    return vecfile


def train_test_data(datafile):
    data = pd.read_csv(datafile)
    '''print("------------------")
    print(data.columns)
    print("------------------")
    print("垃圾邮件分类的数据的维度:{}".format(data.shape))
    gs = data.groupby('label')
    print(gs.size())'''
    x = data.drop('label', axis=1)  # 去掉label列
    y = data.label
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)  # 训练集:测试集=8:2,按照y中的比例分配
    '''print("------------------")
    print("train data: ", y_train.shape[0])
    print("test data: ", y_test.shape[0])'''
    return x_train, x_test, y_train, y_test

def preprocessing(x, y):
    x = np.array(x)
    y = np.array(y)
    d = x.shape[1]
    count = np.zeros(2*d+1, dtype=int).tolist()
    pr = np.zeros(2*d+1, dtype=int).tolist()
    total = y.shape[0]

    for index in range(total):
        if (y[index] == 0):  # y=0
            count[0] += 1
            for j in range(d):
                if (x[index][j] == 0):  # xj=0|y=0
                    count[j + 1] += 1
        else:  # y=1
            for j in range(d):
                if (x[index][j] == 0):  # xj=0|y=1
                    count[j + d + 1] += 1
    # 0 Pr(y=0);      2d+1 Pr(y=1)=1-Pr(y=0)
    # 1 Pr(x0=0|y=0); 2d+2 Pr(x0=1|y=0)=1-Pr(x0=0|y=0)
    # 2 Pr(x1=0|y=0); 2d+3 Pr(x1=1|y=0)=1-Pr(x1=0|y=0)
    # ...... d        3d+1
    # d+1 Pr(x0=0|y=1); 3d+2 Pr(x0=1|y=1)=1-Pr(x0=0|y=1)
    # d+2 Pr(x1=0|y=1); 3d+3 Pr(x1=1|y=1)=1-Pr(x1=0|y=1)
    #...... 2d         4d+1

    pr[0] = (count[0] + 1) / (total + 2)  #拉普拉斯修正
    #y=0的个数count[0],y=1的个数total-count[0]
    for i in range(1,d+1):
        pr[i] = (count[i]+1) / (count[0]+2)
    for i in range(d+1,2*d+1):
        pr[i] = (count[i]+1) / (total-count[0]+2)

    '''print("------------------")
    print("preprocessing pr:", pr)'''
    return pr

def prediction(pr, x):  # 预测一个d维向量的label
    d=len(x)
    pry0 = pr[0] #y=0
    pry1 = 1-pr[0] #y=1
    for i in range(d):
        if (x[i] == 0):
            pry0 *= pr[i+1] #xi=0|y=0
            pry1 *= pr[i+d+1] #xi=0|y=1
        else:
            pry0 *=1-pr[i+1] #xi=1|y=0
            pry1 *=1-pr[i+d+1] #xi=1|y=1

    if(pry1>pry0): #spam
        return 1
    else:
        return 0

def test(x,pr): #预测test数据集的labels
    x=np.array(x)
    y_prediction=[]
    for i in range(x.shape[0]):
        y_prediction.append(prediction(pr,x[i]))
    '''print("------------------")
    print("y_prediction:",y_prediction)'''
    return y_prediction

def assessment(y_prediction,y):
    y=y.tolist()
    epsilon=0.00001
    TP,TN,FP,FN=0,0,0,0
    for i in range(len(y)):
        if y[i]==1 and y_prediction[i]==1:
            TP+=1
        elif y[i]==0 and y_prediction[i]==0:
            TN+=1
        elif y[i]==0 and y_prediction[i]==1:
            FP+=1
        else:
            FN+=1
    '''print("------------------")
    print("TP:", TP, "FN:", FN, "FP:", FP, "TN:", TN)'''
    Accuracy=(TP+TN)/(TP+TN+FP+FN+epsilon)
    Precision=TP/(TP+FP+epsilon)
    Recall=TP/(TP+FN+epsilon)
    F_Score=(2*Precision*Recall)/(Precision+Recall+epsilon)

    '''print("Accuracy:",Accuracy,"Precision:",Precision,"Recall:",Recall,"F_Score:",F_Score)'''
    return Accuracy,Precision,Recall,F_Score

def Model(feature_d):
    filename='data/label_vec_'+str(feature_d)+'.csv'
    datafile=generate_fecturefile(words_tfidf,tokens_list_of_list,feature_d,filename) #生成数据
    x_train, x_test, y_train, y_test = train_test_data(datafile) #切分数据
    pr = preprocessing(x_train, y_train)  #由train数据，计算pr
    y_prediction = test(x_test, pr) #预测test数据
    Accuracy,Precision,Recall,F_Score=assessment(y_prediction, y_test) #模型评估
    return Accuracy,Precision,Recall,F_Score

def picture(d_s,epoch):
    plt.figure(figsize=(12,6))  # 创建绘图对象
    x = []
    y = []
    average_s = []
    average_acc = []
    for d in d_s:
        average = 0
        acc=0
        for i in range(epoch):
            Accuracy, Precision, Recall, F_Score = Model(d)
            x.append(d)
            y.append(F_Score)
            acc+=Accuracy
            average += F_Score
        average /= epoch
        acc /= epoch
        average_acc.append(acc)
        average_s.append(average)
    plt.plot(x, y, color='c',marker='o', markersize=2, linewidth=0)
    plt.plot(d_s, average_s, color='c', marker='o', linestyle='--', markersize=3, linewidth=1,label='average F_Score')
    for a, b in zip(d_s, average_s):
        plt.text(a, b, (b), ha='left', va='bottom', fontsize=9)
    plt.plot(d_s, average_acc, color='g', marker='o', linestyle='-.', markersize=3, linewidth=1, label='average Accuarcy')
    for a, b in zip(d_s, average_acc):
        plt.text(a, b, (b), ha='left', va='bottom', fontsize=9)

    x_major_locator = MultipleLocator(2)
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()# ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)# x轴的主刻度设置
    ax.yaxis.set_major_locator(y_major_locator)# y轴的主刻度设置
    plt.xlim(5,17)
    plt.ylim(-0.05,1.05)
    plt.xlabel("feature_d", fontsize='15')  # X轴标签
    plt.ylabel("F_Score", fontsize='15')  # Y轴标签
    plt.legend(fontsize='15',loc="upper right")
    plt.savefig("data/d_FScore_Accuracy.png") #保存图
    plt.show()  # 显示图



feature_ds=[6,8,10,12,14,16]
epoch=100
picture(feature_ds,epoch)