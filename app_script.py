#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing all dependencies
from func.classes_and_functions import Read_train


# In[2]:


n_users,n_posts,train,test= Read_train.read_data('self')
trainn,testt = Read_train.user_post_ids('self')
model = Read_train.get_model()


# In[4]:


data =Read_train.get_estimation_data()


# In[ ]:


from flask import Flask,request
import pandas as pd
app = Flask(__name__)
 
@app.route("/",methods =['GET','POST'])
def select():
    
    user=request.args.get('nm','')
    
    user = int(user)
    
    selected = data.loc[data['user_id']==user]

    return selected.to_json()
 
if __name__ == "__main__":
    app.run()


# In[ ]:




