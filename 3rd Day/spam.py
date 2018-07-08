# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:09:24 2018

@author: OpenSource
"""
import string
translator=str.maketrans('','',string.punctuation)
text=open("SMSSpamCollection","r")
line=text.readlines()

cls={'ham':dict(),'spam':dict()}
for line in line:
    words=line.split();
    for word in words:
        if word not in cls[words[0]].keys():
            cls[words[0]].update({word:1})
        else:
            cls[words[0]][word]=cls[words[0]][word]+1
            
    for opposite in cls.keys():
        if(opposite !=words[0]):
            if word not in cls[opposite].keys():
                cls[opposite].update({word:1})
        
        
total_ham=sum(cls['ham'].values())
total_spam=sum(cls['spam'].values())

for val in cls:
    total=sum(cls[val].values())
    for val1 in cls[val]:
            cls[val][val1]=cls[val][val1]/total 
            
    print("Training is done://---->go for next step")
    user_input=input('Enter input word:')
    user_input=user_input.translate(translator)
    words1=user_input.split()

    total_words=len(cls['ham'])+len(cls['spam'])

    #probability
    p_spam=len(cls['spam'])/total_words
    p_ham=len(cls['ham'])/total_words

    predict_spam=p_spam
    predict_ham=p_ham
    
    for word in words1:
        predict_spam=predict_spam * cls['spam'][word]
        predict_ham=predict_ham * cls['ham'][word]

    if(predict_ham>predict_spam):   
        print("not spam")
        print("probabilty:{}".format(predict_ham))
    else:
        print("Spam")
        print("probabilty:{}".format(predict_spam))
        text.close()