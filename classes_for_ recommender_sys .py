#!/usr/bin/env python
# coding: utf-8



#importing dependencies
from keras.models import load_model
from keras import regularizers
import pandas as pd
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout
from keras.layers.merge import Dot, multiply, concatenate
from keras.utils import np_utils
from keras.utils.data_utils import get_file
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import skipgrams
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split


# In[ ]:


#preprocessing data & split
class Read_train:
    def read_data(self):
        df = pd.read_csv('C:/Users/faiza/OneDrive/Desktop/Post_activity.csv', sep ='|')
        df.loc[df.Likes>0,'Likes']=1
        df.loc[df.Comments>0,'Comments']=1
        df.loc[df.Shares>0,'Shares']=1
        df.loc[df.Downloads>0,'Downloads']=1
        df.loc[df.Views>0,'Views']=1

        df['Rating']=df['Likes']+df['Comments']+df['Shares']+df['Downloads']+df['Views']
        df.drop(['Likes','Comments','Shares','Downloads','Views'],axis=1,inplace=True)
    
        df.UserId = df.UserId.astype('category').cat.codes.values
        df.PostId = df.PostId.astype('category').cat.codes.values

        train,test =train_test_split(df, test_size = 0.1,random_state = 42 )

        n_users = len(df.UserId.unique()) 
        n_posts = len(df.PostId.unique())
        return n_users,n_posts,train,test

    
    
    def user_post_ids(self):
        dff = pd.read_csv('C:/Users/faiza/OneDrive/Desktop/Post_activity.csv', sep ='|')
        dff.loc[dff.Likes>0,'Likes']=1
        dff.loc[dff.Comments>0,'Comments']=1
        dff.loc[dff.Shares>0,'Shares']=1
        dff.loc[dff.Downloads>0,'Downloads']=1
        dff.loc[dff.Views>0,'Views']=1

        dff['Rating']=dff['Likes']+dff['Comments']+dff['Shares']+dff['Downloads']+dff['Views']
        dff.drop(['Likes','Comments','Shares','Downloads','Views'],axis=1,inplace=True)
        trainn,testt = train_test_split(dff,test_size = 0.1,random_state =42)
        
        return trainn, testt


    def define_model(self, n_users,n_posts,train,test):
        post_input = Input(shape=[1], name="post-Input")
        post_embedding = Embedding(n_posts+1,10,  name="post-Embedding")(post_input)
        lp = Dense(10,activation = 'relu',kernel_regularizer=regularizers.l2(0.001),)(post_embedding)
        Dropout(0.4)
        post_vec = Flatten(name="Flatten-post")(lp)

        user_input = Input(shape=[1], name="User-Input")
        user_embedding = Embedding(n_users+1, 10, name="User-Embedding")(user_input)
        l2 = Dense(10,activation = 'relu',kernel_regularizer=regularizers.l2(0.001))(user_embedding)
        Dropout(0.4)
        user_vec = Flatten(name="Flatten-Users")(l2)

        product_layer = Dot(name="Dot-Product", axes=1)([post_vec, user_vec])

        fully_connected_layer = Dense(10,activation ='relu')(product_layer)
        fully_connected_layer_2 = Dense(10,activation ='relu')(fully_connected_layer)
        fully_connected_layer_3 = Dense(10,activation ='relu')(fully_connected_layer_2)
        fully_connected_layer_4 = Dense(10,activation ='relu')(fully_connected_layer_3)


        output_connected_layer = Dense(1,activation ='linear')(fully_connected_layer_4)

        model = Model([user_input, post_input],output_connected_layer)
        model.compile(loss='mse', optimizer='adam', metrics=["mae"])
        return model
    
    def train_model():
        model =Read_train.define_model('self', n_users,n_posts,train,test)
        history = model.fit([train.UserId, train.PostId], train.Rating,validation_split=0.1 , epochs= 3, verbose=1)
        model.save('recommender_model.h5')
        return history
    
    def get_model():
        global model
        model = load_model('recommender_model.h5')
        print('model loaded')
        return model
    
    
    def get_estimation_data():
        def duplicate(testList,n ): 
            return list(testList*n)
                
        n_users,n_posts,train,test=Read_train.read_data('self')
        trainn,testt=Read_train.user_post_ids('self')
        len_post = len(test.PostId.unique())
        len_user= len(testt.UserId.unique())
        p = test.PostId.unique()
        unique_postids = p.tolist()
        upids=duplicate(unique_postids,len_user) #post_ids_looped


        u =test.UserId.unique()
        unique_userids =u.tolist()
        un = np.array(unique_userids)
        user_loop =np.repeat(unique_userids,len_post) #user_ids_looped
        ttpids = testt['PostId'].unique()
        ttuid = testt['UserId'].unique()
        pp = testt.PostId.unique()
        uunique_postids = pp.tolist()
        uupids=duplicate(uunique_postids,len_user) #post_ids_looped


        uu =testt.UserId.unique()
        uunique_userids =uu.tolist()
        uun = np.array(uunique_userids)
        uuser_loop =np.repeat(uunique_userids,len_post) #user_ids_looped
        post_data = np.array(upids)
        user = np.array(user_loop)
        estimations = model.predict([user, post_data]) #predictions
   
        pid =pd.DataFrame(uupids)  #forming dataframes
        uid =pd.DataFrame(uuser_loop)
        estimation =pd.DataFrame(estimations)
        dataa = pd.merge(estimation,pid,left_index =True,right_index = True)
        data = pd.merge(dataa,uid,left_index = True, right_index= True)
        data.rename(columns={'0_x':'estimation','0_y':'post_id',0:'user_id'},inplace = True)
        final_data_sorted = data.groupby(["user_id"]).apply(lambda x: x.sort_values(["estimation"], ascending = False)).reset_index(drop=True)
        return final_data_sorted

   
       
       
        
    






