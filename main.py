# main.py
import os
import base64
import io
import math
from flask import Flask, flash, render_template, Response, redirect, request, session, abort, url_for
import mysql.connector
import hashlib
import datetime
from datetime import datetime
from datetime import date
import random
import textwrap
from urllib.request import urlopen
import webbrowser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import urllib.request
import urllib.parse

####
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import plotly.express as px #interactive visualizations
import seaborn as sns

import string
#import nltk
#nltk.download()
##nltk.download('punkt')
##nltk.download('stopwords')
##nltk.download('wordnet')
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer 
from wordcloud import WordCloud, STOPWORDS
#from nltk.tokenize import word_tokenize

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

from gensim.parsing.preprocessing import remove_stopwords, STOPWORDS
from gensim.parsing.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Import tf and keras - as embedded in tensorflow!
'''import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Model'''
###

import re
#from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
from sklearn.linear_model import LogisticRegressionCV


import matplotlib.pyplot as plt
#import tensorflow as tf

#from tensorflow.keras.preprocessing.text import Tokenizer
#import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  charset="utf8",
  database="fake_news_classify"

)
app = Flask(__name__)
##session key
app.secret_key = 'abcdef'
#######
UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = { 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#####
@app.route('/', methods=['GET', 'POST'])
def index():
    msg=""
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM register WHERE uname = %s AND pass = %s AND dstatus=0', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('userhome'))
        else:
            msg = 'Incorrect username/password! or Your Account has blocked!'
    

    return render_template('index.html',msg=msg)



@app.route('/login', methods=['GET', 'POST'])
def login():
    msg=""

    
    if request.method=='POST':
        uname=request.form['uname']
        pwd=request.form['pass']
        cursor = mydb.cursor()
        cursor.execute('SELECT * FROM admin WHERE username = %s AND password = %s', (uname, pwd))
        account = cursor.fetchone()
        if account:
            session['username'] = uname
            return redirect(url_for('admin'))
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html',msg=msg)

@app.route('/register', methods=['GET', 'POST'])
def register():
    msg=""
    mycursor = mydb.cursor()
    mycursor.execute("SELECT max(id)+1 FROM register")
    maxid = mycursor.fetchone()[0]
    if maxid is None:
        maxid=1
    if request.method=='POST':
        name=request.form['name']
        mobile=request.form['mobile']
        email=request.form['email']
        uname=request.form['uname']
        pass1=request.form['pass']
        
        cursor = mydb.cursor()

        now = datetime.now()
        rdate=now.strftime("%d-%m-%Y")
    
        sql = "INSERT INTO register(id,name,mobile,email,uname,pass) VALUES (%s,%s, %s, %s, %s, %s)"
        val = (maxid,name,mobile,email,uname,pass1)
        cursor.execute(sql, val)
        mydb.commit()            
        print(cursor.rowcount, "Registered Success")
        result="sucess"
        if cursor.rowcount==1:
            return redirect(url_for('index'))
        else:
            msg='Already Exist'
    return render_template('/register.html',msg=msg)

@app.route('/userhome', methods=['GET', 'POST'])
def userhome():
    msg=""
    cnt=0
    data2=[]
    uname=""
    mess=""
    act=""
    st=""
    s1=""
    msg1=""
    
    if 'username' in session:
        uname = session['username']
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data = mycursor.fetchone()

    today = date.today()
    rdate = today.strftime("%d-%m-%Y")
    ########
    if request.method == 'POST':
        title= request.form['title']
        post= request.form['news']
        x=0
        y=0
        deptype=0
        f1=open("lexicon.txt","r")
        dat=f1.read()
        f1.close()
        dat1=dat.split("|")
        for rd in dat1:
           
            t1=post
            t2=rd.strip()
            if t2 in t1:
                act="yes"
                st=1
                x+=1
                break
        ###########
        f11=open("true1.txt","r")
        dat1=f11.read()
        f11.close()
        dat11=dat1.split("|")
        for rd1 in dat11:
           
            t11=post
            t21=rd1.strip()
            if t21 in t11:
                act="yes"
                st=1
                y+=1
                break
        ###########
        if x>0:
            act="1"
            st="1"
            mess="Fake News!!"
        elif y>0:
            act="1"
            st="3"
            mess="True News!!"
        else:
            act="1"
            st="2"
            mess="Normal News"

        mycursor.execute("SELECT count(*) FROM user_post where uname=%s && status=1",(uname,))
        cnt = mycursor.fetchone()[0]
        if cnt==2:
            msg1="warn"
        elif cnt==3:
            msg1="block"
            mycursor.execute("update register set dstatus=1 where uname=%s",(uname,))
            mydb.commit()

        mycursor.execute("SELECT max(id)+1 FROM user_post")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
        
        sql = "INSERT INTO user_post (id,uname,title,text_post,rdate,status) VALUES(%s,%s,%s,%s,%s,%s)"
        val = (maxid,uname,title,post,rdate,st)
        mycursor.execute(sql,val)
        print(sql,val)
        mydb.commit()

        if st=="2" or st=="3":
            msg="success"
        else:
            msg="fail"

    mycursor.execute("SELECT count(*) FROM user_post where status>1 order by id desc")
    cnt = mycursor.fetchone()[0]

    #my_wrap = textwrap.TextWrapper(width = 20)
    
    if cnt>0:
        s1="1"
        mycursor.execute("SELECT * FROM user_post where status>1 order by id desc")
        dat2 = mycursor.fetchall()
        for dat in dat2:
            dt=[]
            dt.append(dat[0])
            dt.append(dat[1])
            dt.append(dat[2])

            #single_line=dat[3]
            #dd=my_wrap.fill(text = single_line)
            dt.append(dat[3])
            dt.append(dat[4])
            dt.append(dat[5])
            dt.append(dat[6])
            data2.append(dt)

    return render_template('userhome.html',msg=msg,msg1=msg1,data=data,mess=mess,act=act,st=st,s1=s1,data2=data2)

@app.route('/user_post', methods=['GET', 'POST'])
def user_post():
    st=0
    uname=""
    mess=""
    cnt=0
    act=""
    file_name=""
    if 'username' in session:
        uname = session['username']
    cursor = mydb.cursor()
    cursor.execute('SELECT * FROM register WHERE uname = %s', (uname, ))
    data = cursor.fetchone()

    pcursor = mydb.cursor()
    pcursor.execute('SELECT * FROM user_post u,register r where u.uname=r.uname order by u.id desc')
    pdata = pcursor.fetchall()

    pcursor1 = mydb.cursor()
    pcursor1.execute('SELECT count(*) FROM user_post WHERE uname = %s and status=1', (uname, ))
    cnt = pcursor1.fetchone()[0]
    print(cnt)
    
    if request.method=='GET':
        act = request.args.get('act')
    if request.method == 'POST':
        post= request.form['message']
        if 'file' not in request.files:
            flash('No file Part')
            return redirect(request.url)
        file= request.files['file']

        mycursor = mydb.cursor()
        mycursor.execute("SELECT max(id)+1 FROM user_post")
        maxid = mycursor.fetchone()[0]
        if maxid is None:
            maxid=1
            
        if file.filename == '':
            flash('No Select file')
            #return redirect(request.url)
        if file:
            fname = "P"+str(maxid)+file.filename
            file_name = secure_filename(fname)
            
            file.save(os.path.join(app.config['UPLOAD_FOLDER']+"/comments/", file_name))
            
        today = date.today()
        rdate = today.strftime("%d-%m-%Y")

        cursor2 = mydb.cursor()
        
        

        
        sql = "INSERT INTO user_post (id,uname,text_post,photo,rdate,status) VALUES(%s,%s,%s,%s,%s,%s)"
        val = (maxid,uname,post,file_name,rdate,st)
        mycursor.execute(sql,val)
        print(sql,val)
        mydb.commit()
        msg="Upload success"
        return redirect(url_for('user_post',act=act))  
    
    return render_template('user_post.html',data=data,act=act,pdata=pdata)



@app.route('/edit_profile', methods=['GET', 'POST'])
def edit_profile():
    uname=""
    if 'username' in session:
        uname = session['username']
        print(uname)    
    mycursor = mydb.cursor()
    mycursor.execute('SELECT * FROM register WHERE uname = %s', (uname, ))
    data = mycursor.fetchone()
    
    
    if request.method=='POST':
        name = request.form['name']
        dob = request.form['dob']
        contact = request.form['mobile']
        email = request.form['email']
        location = request.form['location']
        profession = request.form['profession']
        aadhar = request.form['aadhar']
        #filename=('uname.txt')
        #fileread=open(filename,"r+")
        #uname=fileread.read()
        #fileread.close()
        
        sql=("update register set name=%s, dob=%s,mobile=%s,email=%s,location=%s,profession=%s,aadhar=%s,status=1 where uname=%s")
        val=(name,dob, contact, email, location, profession,aadhar, uname)
        mycursor.execute(sql,val)
        mydb.commit()
        print(val)
        msg="success"
        return redirect(url_for('userhome',msg=msg))
    return render_template('edit_profile.html',data=data)


@app.route('/change_profile', methods=['GET', 'POST'])
def change_profile():
    uid=""
    uname=""
    print(uid)
    if 'username' in session:
        uname = session['username']
    print(uname)

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM register where uname=%s",(uname,))
    data = mycursor.fetchone()
    
    if request.method=='GET':
        act = request.args.get('act')
        uid = request.args.get('uname')
        
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file Part')
            return redirect(request.url)
        file= request.files['file']
        print(file)
        if file.filename == '':
            flash('No Select file')
            return redirect(request.url)
        if file:
            fname = file.filename
            fimg = uname+".png"
            file_name = secure_filename(fimg)
            print(file_name)
            file.save(os.path.join(app.config['UPLOAD_FOLDER']+"/photo/", file_name))
            
            
            mycursor.execute("update register set photo=1 where uname=%s", (uname, ))
            mydb.commit()
            msg="Upload success"
            return redirect(url_for('userhome'))  
    
    return render_template('change_profile.html',data=data)



############################################
@app.route('/admin', methods=['GET', 'POST'])
def admin():
    msg=""
    
    
    return render_template('admin.html',msg=msg)



@app.route('/process1', methods=['GET', 'POST'])
def process1():
    msg=""

    #df_data = pd.read_csv("dataset/news_data.csv")
    df_true = pd.read_csv("dataset/True1.csv")
    df_fake = pd.read_csv("dataset/Fake1.csv")

    # Show top 5 rows in df_true
    dat=df_true.head()
    data1=[]
    for ss1 in dat.values:
        data1.append(ss1)

    # Show top 5 rows in df_fake
    dat2=df_fake.head()
    data2=[]
    for ss2 in dat2.values:
        data2.append(ss2)


    print(df_true.shape)
    print("-"*20)
    print(df_fake.shape)

    tr=df_true.shape
    fk=df_fake.shape

    print(df_true.dtypes)
    print("-"*20)
    print(df_fake.dtypes)

    # Check if there is any missing data in True dataframe
    print(df_true.isnull().sum())

    print("-"*20)

    # Check if there is any missing data in Fake dataframe
    print(df_fake.isnull().sum())


    # Show a list of articles considered as real news
    dat3=df_true.describe()
    arr=['count','unique','top','freq']
    data3=[]
    i=0
    for ss3 in dat3.values:
        dt=[]
        dt.append(arr[i])
        dt.append(ss3)
        data3.append(dt)
        i+=1

    # how a list of articles considered as fake news
    dat4=df_true.describe()
    arr2=['count','unique','top','freq']
    data4=[]
    i=0
    for ss4 in dat4.values:
        dt=[]
        dt.append(arr2[i])
        dt.append(ss4)
        data4.append(dt)
        i+=1


    # Add a attribute column "class" to indicate whether the news is real(1) or fake(0)
    df_true['target'] = 1
    df_fake['target'] = 0

    # After adding "class" column in df_true
    dat5=df_true.head()
    data5=[]
    for ss5 in dat5.values:
        data5.append(ss5)

    # After adding "class" column in df_fake
    dat6=df_fake.head()
    data6=[]
    for ss6 in dat6.values:
        data6.append(ss6)


    # Concatenate df_true and df_fake into df_dataset
    df_dataset = pd.concat([df_true, df_fake]).reset_index(drop=True)
    dat7=df_dataset
    data7=[]
    i=0
    for ss7 in dat7.values:
        if i<=20:
            data7.append(ss7)

        i+=1

    #Combining the title and text columns
    df_dataset['text'] = df_dataset['title'] + " " + df_dataset['text']

    #Deleting few columns from the data 
    del df_dataset['title']
    del df_dataset['subject']
    del df_dataset['date']

    # Review content of text in first row
    data8=df_dataset['text'][0]


    return render_template('process1.html',msg=msg,data1=data1,data2=data2,tr=tr,fk=fk,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7,data8=data8)

######
@app.route('/process2', methods=['GET', 'POST'])
def process2():
    msg=""

    df_true = pd.read_csv("dataset/True3.csv")
    df_fake = pd.read_csv("dataset/Fake3.csv")

    # Show top 5 rows in df_true
    df_true.head()
    

    # Show top 5 rows in df_fake
    df_fake.head()
    
    #print(df_true.shape)
    #print("-"*20)
    #print(df_fake.shape)

    tr=df_true.shape
    fk=df_fake.shape

    #print(df_true.dtypes)
    #print("-"*20)
    #print(df_fake.dtypes)

    # Check if there is any missing data in True dataframe
    print(df_true.isnull().sum())

    #print("-"*20)

    # Check if there is any missing data in Fake dataframe
    print(df_fake.isnull().sum())


    # Show a list of articles considered as real news
    df_true.describe()
    

    # how a list of articles considered as fake news
    df_true.describe()
    

    # Add a attribute column "class" to indicate whether the news is real(1) or fake(0)
    df_true['target'] = 1
    df_fake['target'] = 0

    # After adding "class" column in df_true
    df_true.head()
    
    # After adding "class" column in df_fake
    df_fake.head()
    
    # Concatenate df_true and df_fake into df_dataset
    df_dataset = pd.concat([df_true, df_fake]).reset_index(drop=True)
    df_dataset
    

    #Combining the title and text columns
    df_dataset['text'] = df_dataset['title'] + " " + df_dataset['text']

    #Deleting few columns from the data 
    del df_dataset['title']
    del df_dataset['subject']
    del df_dataset['date']

    # Review content of text in first row
    df_dataset['text'][0]
    ################################
    #Data Cleaning
    #Choosing the language as english
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",',','.','I','\'','-','/']
    
    #stop = set(stop_words.words('english'))

    #Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        text = text.lower()
        for i in text.split():
            if i.strip() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    #Removing the noisy text
    def clean_text(text):
        text = remove_stopwords(text)
        return text

    df_dataset['text'] = df_dataset['text'].apply(clean_text)
    data1=df_dataset['text'][0]

    # Removing words under 2 or less characters from text!
    def text_preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if len(token) > 3:
                result.append(token)       
        return result

    df_dataset['clean'] = df_dataset['text'].apply(text_preprocess)

    # Show cleaned up news after removing stopwords
    data2=df_dataset['clean'][0]

    dat3=df_dataset.head()
    data3=[]
    for ss3 in dat3.values:
        data3.append(ss3)
    
    # Collect total vocabulary
    lst_words = []
    for i in df_dataset['clean']:
        for j in i:
            lst_words.append(j)
            
    # From first row to the last row, all words are stored into lst_words  
    data4=len(lst_words)
    ##
    # Derive the number of "UNIQUE" words using set()
    uni_words = len(list(set(lst_words)))
    print(f"The maximum number of words in a dictionary is: {uni_words}")
    data5=uni_words

    ##
    # Joined elements of df_dataset['clean'] into a new column (for training Machine Learning models)
    df_dataset['clean_joined'] = df_dataset['clean'].apply(lambda x: " ".join(x))
    dat6=df_dataset.head()
    data6=[]
    for ss6 in dat6.values:
        data6.append(ss6)

    data7=df_dataset['clean_joined'][0]

    return render_template('process2.html',msg=msg,data1=data1,data2=data2,data3=data3,data4=data4,data5=data5,data6=data6,data7=data7)
######
@app.route('/process3', methods=['GET', 'POST'])
def process3():
    msg=""

    df_true = pd.read_csv("dataset/True1.csv")
    df_fake = pd.read_csv("dataset/Fake1.csv")

    # Show top 5 rows in df_true
    df_true.head()
    

    # Show top 5 rows in df_fake
    df_fake.head()
    
    #print(df_true.shape)
    #print("-"*20)
    #print(df_fake.shape)

    tr=df_true.shape
    fk=df_fake.shape

    #print(df_true.dtypes)
    #print("-"*20)
    #print(df_fake.dtypes)

    # Check if there is any missing data in True dataframe
    print(df_true.isnull().sum())

    #print("-"*20)

    # Check if there is any missing data in Fake dataframe
    print(df_fake.isnull().sum())


    # Show a list of articles considered as real news
    df_true.describe()
    

    # how a list of articles considered as fake news
    df_true.describe()
    

    # Add a attribute column "class" to indicate whether the news is real(1) or fake(0)
    df_true['target'] = 1
    df_fake['target'] = 0

    # After adding "class" column in df_true
    df_true.head()
    
    # After adding "class" column in df_fake
    df_fake.head()
    
    # Concatenate df_true and df_fake into df_dataset
    '''df_dataset = pd.concat([df_true, df_fake]).reset_index(drop=True)
    df_dataset
    

    #Combining the title and text columns
    df_dataset['text'] = df_dataset['title'] + " " + df_dataset['text']

    #Deleting few columns from the data 
    del df_dataset['title']
    del df_dataset['subject']
    del df_dataset['date']

    # Review content of text in first row
    df_dataset['text'][0]
    ################################
    #Data Cleaning
    #Choosing the language as english
    stop = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't",',','.','I','\'','-','/']
    
    #stop = set(stopwords.words('english'))

    #Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        text = text.lower()
        for i in text.split():
            if i.strip() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    #Removing the noisy text
    def clean_text(text):
        text = remove_stopwords(text)
        return text

    df_dataset['text'] = df_dataset['text'].apply(clean_text)
    data1=df_dataset['text'][0]

    # Removing words under 2 or less characters from text!
    def text_preprocess(text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if len(token) > 3:
                result.append(token)       
        return result

    df_dataset['clean'] = df_dataset['text'].apply(text_preprocess)

    # Show cleaned up news after removing stopwords
    df_dataset['clean'][0]

    df_dataset.head()
    
    # Collect total vocabulary
    lst_words = []
    for i in df_dataset['clean']:
        for j in i:
            lst_words.append(j)
            
    # From first row to the last row, all words are stored into lst_words  
    len(lst_words)
    ##
    # Derive the number of "UNIQUE" words using set()
    uni_words = len(list(set(lst_words)))
    print(f"The maximum number of words in a dictionary is: {uni_words}")
    uni_words

    ##
    # Joined elements of df_dataset['clean'] into a new column (for training Machine Learning models)
    df_dataset['clean_joined'] = df_dataset['clean'].apply(lambda x: " ".join(x))
    df_dataset.head()
    
    df_dataset['clean_joined'][0]
    ############

    #Checking for imbalance in the dataset
    count = df_dataset['target'].value_counts().values
    sns.barplot(x = [0, 1], y = count)
    plt.title('target variable count')
    #plt.savefig("static/graph/graph1.png")
    plt.close()

    #Word cloud for real news
    cloud = WordCloud(max_words = 1800, 
                      width = 1600, 
                      height = 800,
                      stopwords = STOPWORDS, 
                      background_color = "white").generate(" ".join(df_dataset[df_dataset['target'] == 1].text))
    plt.figure(figsize=(40, 30))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.show()
    #plt.savefig("static/graph/graph2.png")
    plt.close()

    #Word cloud for fake news
    cloud = WordCloud(max_words = 1800, 
                      width = 1600, 
                      height = 800,
                      stopwords = STOPWORDS, 
                      background_color = "white").generate(" ".join(df_dataset[df_dataset['target'] == 0].text))
    plt.figure(figsize=(40, 30))
    plt.imshow(cloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    #plt.show()
    #plt.savefig("static/graph/graph3.png")
    plt.close()'''

    return render_template('process3.html',msg=msg)



##Fake News Classification using LogReg & LSTM
@app.route('/process4', methods=['GET', 'POST'])
def process4():
    msg=""
    data1=[]
    fake_df = pd.read_csv('dataset/Fake.csv')
    real_df = pd.read_csv('dataset/True.csv')

    fake_df = fake_df[['title', 'text']]
    real_df = real_df[['title', 'text']]

    fake_df['class'] = 0
    real_df['class'] = 1
    df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
    df.head(10)
    df.shape
    df['title_text'] = df['title'] + ' ' + df['text']
    df.drop(['title', 'text'], axis=1, inplace=True)
    df.head()

    '''def preprocessor(text):

        text = re.sub('<[^>]*>', '', text)
        text = re.sub(r'[^\w\s]','', text)
        text = text.lower()

        return text

    df['title_text'] = df['title_text'].apply(preprocessor)

    porter = PorterStemmer()

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    #TF-IDF
    tfidf = TfidfVectorizer(strip_accents=None,
                            lowercase=False,
                            preprocessor=None,
                            tokenizer=tokenizer_porter,
                            use_idf=True,
                            norm='l2',
                            smooth_idf=True
                           )
    X = tfidf.fit_transform(df['title_text'])
    y = df['class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    ###import warnings
    ###warnings.filterwarnings("ignore")

    clf = LogisticRegressionCV(cv=5, scoring='accuracy', random_state=0, n_jobs=-1, verbose=2, max_iter=300).fit(X_train, y_train)

    ####Evaluate the performance.

    clf.score(X_test, y_test)'''

    ###from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
    #y_pred = clf.predict(X_test)
    #print("Accuracy with Logreg: {}".format(accuracy_score(y_test, y_pred)))
    #print(classification_report(y_test, y_pred))

    '''binary_predictions = []

    for i in y_pred:
        if i >= 0.5:
            binary_predictions.append(1)
        else:
            binary_predictions.append(0)'''

    #import matplotlib.pyplot as plt
    #import seaborn as sns
    '''matrix = confusion_matrix(binary_predictions, y_test, normalize='all')
    plt.figure(figsize=(10, 6))
    ax= plt.subplot()
    sns.heatmap(matrix, annot=True, ax = ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted Labels', size=15)
    ax.set_ylabel('True Labels', size=15)
    ax.set_title('Confusion Matrix', size=15)
    ax.xaxis.set_ticklabels([0,1], size=15)
    ax.yaxis.set_ticklabels([0,1], size=15);

    #plt.savefig("static/graph/graph4.png")
    plt.close()'''


    #Using LSTM
    
    #plt.style.use('ggplot')
    #print("Tensorflow version " + tf.__version__)

    # loading the data again
    #fake_df = pd.read_csv('dataset/Fake.csv')
    #real_df = pd.read_csv('dataset/True.csv')

    fake_df = fake_df[['title', 'text']]
    real_df = real_df[['title', 'text']]

    fake_df['class'] = 0
    real_df['class'] = 1

    '''plt.figure(figsize=(10, 5))
    plt.bar('Fake News', len(fake_df), color='orange')
    plt.bar('Real News', len(real_df), color='green')
    plt.title('Distribution of Fake News and Real News', size=12)
    plt.xlabel('News Type', size=12)
    plt.ylabel('# of News Articles', size=12);
    #plt.savefig("static/graph/graph5.png")
    plt.close()

    df = pd.concat([fake_df, real_df], ignore_index=True, sort=False)
    dat1=df.head(20)
    data1=[]
    for ss1 in dat1.values:
        data1.append(ss1)
    
    df['title_text'] = df['title'] + ' ' + df['text']
    df.drop(['title', 'text'], axis=1, inplace=True)

    X = df['title_text']
    y = df['class']'''

    '''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    def normalize(data):
        normalized = []
        for i in data:
            i = i.lower()
            # get rid of urls
            i = re.sub('https?://\S+|www\.\S+', '', i)
            # get rid of non words and extra spaces
            i = re.sub('\\W', ' ', i)
            i = re.sub('\n', '', i)
            i = re.sub(' +', ' ', i)
            i = re.sub('^ ', '', i)
            i = re.sub(' $', '', i)
            normalized.append(i)
        return normalized

    X_train = normalize(X_train)
    X_test = normalize(X_test)
    #We put the parameters at the top like this to make it easier to change and edit.

    vocab_size = 10000
    embedding_dim = 64
    max_length = 256
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = '<OOV>'
    '''
    #Tokenization
    ##### tokenizer = Tokenizer(num_words=max_vocab)
    '''tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding=padding_type, truncating=trunc_type, maxlen=max_length)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding=padding_type, truncating=trunc_type, maxlen=max_length)
    '''
    #Building the Model
    '''model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,  return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.summary()

    # We are using early stop, which stops when the validation loss no longer improves.
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=10,validation_split=0.1,verbose=1, batch_size=30, shuffle=True, callbacks=[early_stop])

    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']
    epochs = history.epoch

    plt.figure(figsize=(10,6))
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss', size=15)
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Loss', size=15)
    plt.legend(prop={'size': 15})
    #plt.show()
    plt.savefig("static/graph/graph6.png")
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(epochs, acc, 'g', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy', size=15)
    plt.xlabel('Epochs', size=15)
    plt.ylabel('Accuracy', size=15)
    plt.legend(prop={'size': 15})
    plt.ylim((0.5,1))
    #plt.show()
    plt.savefig("static/graph/graph7.png")
    plt.close()

    model.evaluate(X_test, y_test)

    pred = model.predict(X_test)

    binary_predictions = []

    for i in pred:
        if i >= 0.5:
            binary_predictions.append(1)
        else:
            binary_predictions.append(0)

    print('Accuracy on testing set:', accuracy_score(binary_predictions, y_test))
    print('Precision on testing set:', precision_score(binary_predictions, y_test))
    print('Recall on testing set:', recall_score(binary_predictions, y_test))

    a1=accuracy_score(binary_predictions, y_test)
    a2=precision_score(binary_predictions, y_test)
    a3=recall_score(binary_predictions, y_test)

    matrix = confusion_matrix(binary_predictions, y_test, normalize='all')
    plt.figure(figsize=(10, 6))
    ax= plt.subplot()
    sns.heatmap(matrix, annot=True, ax = ax)

    # labels, title and ticks
    ax.set_xlabel('Predicted Labels', size=15)
    ax.set_ylabel('True Labels', size=15)
    ax.set_title('Confusion Matrix', size=15)
    ax.xaxis.set_ticklabels([0,1], size=15)
    ax.yaxis.set_ticklabels([0,1], size=15);

    plt.savefig("static/graph/graph8.png")
    plt.close()'''
    #,a1=a1,a2=a2,a3=a3
    return render_template('process4.html',msg=msg,data1=data1)

######################################





@app.route('/logout')
def logout():
    # remove the username from the session if it is there
    session.pop('username', None)
    return redirect(url_for('index'))



if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True,host='0.0.0.0', port=5000)


