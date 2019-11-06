from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image, ImageDraw
import PIL, PIL.Image
from io import BytesIO
import base64
import random
import pickle
from bs4 import BeautifulSoup as bsu
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from django.core.mail import send_mail
from django.conf import settings


# Importing Dataset

global data,col_names, group_by,ran_state_values, model, encoded_data 
data = pd.read_pickle('./static/dataset/Friday-WorkingHours-Morning.pcap_ISCX.pkl')
data.columns = data.columns.str.lstrip()
col_names = list(data.columns)
group_by = data.groupby('Label')
with open('./static/dataset/Decision_Tree_Friday-WorkingHours-Morning.pcap_ISCX.model','rb') as f:
    model = pickle.load(f)
with open('./static/dataset/Friday-WorkingHours-Morning.pcap_ISCX_encoded.pkl','rb') as f:
    encoded_data = pickle.load(f)
with open('./static/dataset/file.txt','rb') as f:
    ran_state_values = pickle.load(f)
ran_state_values = list(ran_state_values)


# Create your views here.

def index(request):
    if request.method =="GET":
        return render(request,'index.html')
    elif request.method =="POST":
        name = request.POST.get("name")
        phone = request.POST.get("phone")
        email =[request.POST.get("email")]
        print(name,phone,email)
        feedback = request.POST.get("feedback")
        subject = 'Thanks to Contact Us'
        message = f' Greetings : {name} ,\n It is pleasure to hear from you. Our technical team would reach you soon. Happy Security.'
        email_from = settings.EMAIL_HOST_USER
        send_mail(subject,message,email_from,email)
        our_email = ['your_mail']
        send_mail('Someone Contacted',f'Contacted Person ,\n Name : {name}\n Phone : {phone}\n Message is :  {feedback}',email_from,our_email)
    return render(request,'index.html')
    
def dataset(request):
    return render(request,'./DataSet/index.html')

def visualization(request):
    global data,col_names,group_by 
    if request.method =="GET":
        col_name = col_names[-1]
        data.columns = data.columns.str.lstrip()
        table=group_by[col_name].describe()
        col_name=(list(col_name.split('/')))[0]
        data1 = table.copy()

        data1.drop('count',axis=1,inplace=True)
        if data[col_name].dtype !='object':
            data1.drop('max',axis=1,inplace=True)
            data1.drop('75%',axis=1,inplace=True)
        bar=data1.plot.bar()
        plt.ylabel('Values')
        plt.xlabel(col_name)
        locs, labels = plt.yticks()
        plt.setp(labels,rotation=45)
        locs, labels = plt.xticks()
        plt.setp(labels,rotation=45)
        plt.legend(bbox_to_anchor=(0, 1.05), loc='center left', ncol=7)
        figure=bar.get_figure()
        figure.set_size_inches((8,8))

        table = pd.DataFrame(table)
        table = table.to_html(table_id='vt1')

        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        figure.savefig(f'{col_name}.png')
        plt.close(figure)
        image = Image.open(f'{col_name}.png')
        draw = ImageDraw.Draw(image)
        image.save(buf,'PNG')
        content_type = 'image/png'
        buffercontent = buf.getvalue()
        graphic = base64.b64encode(buffercontent)
        image_ = graphic.decode('utf-8')
        values_to_return = {'tab':table,'cols':col_names,'img':image_}
        return render(request,'./Visualization/index.html',context=values_to_return)
    if request.method =="POST":
        col_name = request.POST.get("col_name")
        data.columns = data.columns.str.lstrip()
        table=group_by[col_name].describe()
        col_name=(list(col_name.split('/')))[0]
        data1 = table.copy()

        data1.drop('count',axis=1,inplace=True)
        if data[col_name].dtype !='object':
            data1.drop('max',axis=1,inplace=True)
            data1.drop('75%',axis=1,inplace=True)
        bar=data1.plot.bar()
        plt.ylabel('Values')
        plt.xlabel(col_name)
        locs, labels = plt.yticks()
        plt.setp(labels,rotation=45)
        locs, labels = plt.xticks()
        plt.setp(labels,rotation=45)
        plt.legend(bbox_to_anchor=(0, 1.05), loc='center left', ncol=7)
        figure=bar.get_figure()
        figure.set_size_inches((8,8))

        table = pd.DataFrame(table)
        table = table.to_html(table_id='vt1')

        buf = io.BytesIO()
        plt.savefig(buf,format='png')
        figure.savefig(f'{col_name}.png')
        plt.close(figure)
        image = Image.open(f'{col_name}.png')
        draw = ImageDraw.Draw(image)
        image.save(buf,'PNG')
        content_type = 'image/png'
        buffercontent = buf.getvalue()
        graphic = base64.b64encode(buffercontent)
        image_ = graphic.decode('utf-8')
        values_to_return = {'tab':table,'cols':col_names,'img':image_}
        return render(request,'./Visualization/index.html',context=values_to_return)

def prediction(request):
    if request.method == "GET":

        quote1='The'
        quote2='will be'
        j=0 # this variable is used for indexing
        ran_state = random.choice(ran_state_values) # this helps for one bot case to come
        to_predict = data.sample(n=5,random_state=ran_state) # choosing any five rows randomly
        list_of_index = list(to_predict.index) # index values will later help to find the same row
        to_predict = to_predict.to_html(table_id='pt') # converting to html code
        to_predict = to_predict.replace('class="dataframe"','class="table table-striped table-responsive"') # replacing with bootstrap class
        to_predict_html = bsu(to_predict,'html.parser') # using beautifulSoup
        val = list_of_index[0] # initially providing the first index value to check box as value

        for i in (to_predict_html.findAll('th')): # finding all the index colums whose values are related to those in list_of_index
            try:
                if (i.string) == str(val):
                    check_box_tag=f' <td><input type="checkbox" name="checkbox" value="{val}" onclick="this.form.submit();"  /></td> '
                    i.replaceWith(bsu(check_box_tag,'html.parser')) # replacing index number with text box with same value
                    j+=1 # updating the index value of the list
                    val = list_of_index[j]
                    
            except Exception as e:
                pass
                
        to_predict = to_predict_html.prettify() # finally converting to html using beautifulSoup
        values_to_return = {'tab':to_predict,'quote1':quote1,'quote2':quote2}
        return render(request,'./Prediction/index.html',context=values_to_return) # returning the dict containing the html code
    if request.method =="POST":

        quote1='This'
        quote2='has been'
        j=0 # this variable is used for indexing
        ran_state = random.choice(ran_state_values) # this helps for one bot case to come
        to_predict = data.sample(n=5,random_state=ran_state) # choosing any five rows randomly
        list_of_index = list(to_predict.index) # index values will later help to find the same row
        to_predict = to_predict.to_html(table_id='pt') # converting to html code
        to_predict = to_predict.replace('class="dataframe"','class="table table-striped table-responsive"') # replacing with bootstrap class
        to_predict_html = bsu(to_predict,'html.parser') # using beautifulSoup
        val = list_of_index[0] # initially providing the first index value to check box as value

        for i in (to_predict_html.findAll('th')): # finding all the index colums whose values are related to those in list_of_index
            try:
                if (i.string) == str(val):
                    check_box_tag=f' <td><input type="checkbox" name="checkbox" value="{val}" onclick="this.form.submit();"  /></td> '
                    i.replaceWith(bsu(check_box_tag,'html.parser')) # replacing index number with text box with same value
                    j+=1 # updating the index value of the list
                    val = list_of_index[j]
                    
            except Exception as e:
                pass
                
        to_predict = to_predict_html.prettify()



        col_num = request.POST.get("checkbox") # getting the index value of the row
        col_num = int(col_num)
        col_to_pred = encoded_data.iloc[col_num] # encoded data is used for the prediction (encoded with label encoder)
        col_to_display = data.iloc[col_num] # and the column to display is taken form the simple data
        right_value = col_to_pred.pop(' Label') # last column is poped from the values to predict
        if right_value.astype(int) == 0:
            right_values = 'Right value was: BENIGN' # the output is manupulated accordingly
        elif right_value.astype(int) == 1:
            right_values = 'Right value was: BOT'

        y_pred = model.predict([col_to_pred])
        y_pred = list(y_pred.astype(int))
        y_pred = int(y_pred[0])
        print(type(y_pred))
        if y_pred == 0:
            y_pred = 'Predicted value is: BENIGN'
        elif y_pred ==1:
            y_pred = 'Predicted value is: BOT'

        col_to_display = pd.DataFrame(col_to_display)
        col_to_display = col_to_display.transpose() # the horizontal dataframe is converted into vertical for better display
        col_to_display = col_to_display.to_html(table_id='pt')
        col_to_display = col_to_display.replace('class="dataframe"','class="table table-striped table-responsive"') # adding bootstrap class
        values_to_return = {'y_pred':y_pred,'right_value':right_values,'tab1':col_to_display,'tab':to_predict,'quote1':quote1,'quote2':quote2} # returning a dict
        return render(request,'./Prediction/index.html',context=values_to_return)

def gallery(request):
    return render(request,'index.html')

def contact(request):
    return render(request,'index.html')

def form(request):
    return render(request,'form.html')

def new(request):
    k=request.POST.get("drop1")
    return HttpResponse(k)

def new1(request):
    k=request.POST.get("exampleName")
    return HttpResponse(k)