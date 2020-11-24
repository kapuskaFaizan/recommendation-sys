from func.classes_and_functions import Read_train
from flask import Flask,request
import pandas as pd

n_users,n_posts,train,test= Read_train.read_data('self')
trainn,testt = Read_train.user_post_ids('self')
model = Read_train.get_model()

data =Read_train.get_estimation_data()

app = Flask(__name__)
 
@app.route("/",methods =['GET','POST'])
def select():
    
    user=request.args.get('nm','')
    
    user = int(user)
    
    selected = data.loc[data['user_id']==user]

    return selected.to_json()
 
if __name__ == "__main__":
    app.run()
