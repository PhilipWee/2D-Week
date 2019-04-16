#IMPORTS FOR KIVY GUI

from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty,StringProperty
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from libdw import pyrebase
import time

#IMPORTS FOR MACHINE LEARNING

import numpy as np
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

#INITIALIZING FIREBASE

url = 'https://twod-dbdd4.firebaseio.com/'
apikey = 'AIzaSyDyw9NUmAGv7alG-bVGzCClfb88MNeqBnc'
config = {"apiKey": apikey,"databaseURL": url,}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
#var = db.child('r_pi').child("temp").get()

#CODE TO READ TEXT FILE

filename = 'NUDES'
f = open('{}.txt'.format(filename),'r')                                         #open the file
data = [['final temp','1s','2s','3s','4s','5s','6s','7s','8s','9s','10s']]      #create an array to store the ALL the data
for line in f:                                                                  #iterate through the lines in the file
    if line.strip() == 'Experiment:':                                           #if the program sees the word "Experiment"
        data_row = []                                                           #create a list for the rows of data
        final_temp_line = f.readline()                                          #read the line with "Experiment in it"
        final_temp_array = final_temp_line.split(" ")                           #organize into an array
        final_temp = final_temp_array[2].strip()                                #organize into an array
        data_row.append(final_temp)                                             #append the final temp in the data row list
        
        for i in range(10):
            temp_line = f.readline()                                            #read the next line
            temp_array = temp_line.split(" ")                                   #organize into an array
            temp_at_time = temp_array[1].strip()                                #organize into an array
            data_row.append(temp_at_time)                                       #append the data into the data_row list
        data.append(data_row)
f.close()


def preprocess(data):
    dataset = data                                                              #save the dataset
    x_data = []                                                                 #split into x and y
    y_data = []                                                                 #split into x and y
    for row_no in range(1,len(dataset)):
        print(dataset[row_no][0])
        y_data.append(dataset[row_no][0])
        x_data_row = []
        for col_no in range(1,len(dataset[0])):
            x_data_row.append(dataset[row_no][col_no])
        x_data.append(x_data_row)
    return x_data,y_data

def prepare_train_test(data,percentage_for_test):                               #write a code to split the data to train and test
    x_data,y_data = preprocess(data)                                            ##split further into testdata and practice data
    x_train,x_test,y_train,y_test = train_test_split(x_data,\
                                                     y_data,\
                                                     test_size = percentage_for_test)
    x_train =  np.array(x_train).astype(float)
    x_test =  np.array(x_test).astype(float)
    y_train =  np.array(y_train).astype(float)
    y_test =  np.array(y_test).astype(float) 
    return x_train,x_test,y_train,y_test

def train_model(data, percentage_for_test):                                     #write a code to train the model, the function should return the trained model
    x_train,x_test,y_train,y_test = prepare_train_test(data,percentage_for_test)
    
    model = linear_model.LinearRegression()                                     #create the linear regression model
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)                                              #predict the y values for a given c
    MSE = mean_squared_error(y_test, y_pred)                                    #calculate the R2,MSE,coef and intercept
    R2 = r2_score(y_test,y_pred)
    coef = model.coef_
    intercept = model.intercept_
    abs_error_vector = np.abs(y_test - y_pred)                                  #generate the results dictionary
    #print('the abs error is ' + str(abs_error_vector))
    results = {"mean squared error" : MSE,
               "intercept" : intercept,
               "coefficients" : coef,
               "r2 score" : R2}
    return results,model

Builder.load_string('''
<MainWidget>:
    GridLayout:
        
        cols: 4
        width:root.width
        height:root.height

        Label:
            text: 'Current Time' 
        Button:
            text: str(round(root.timepassed,1))
            on_press:
                root.begin_twodee()
                root.start_time()
                
        Label: 
            text: 'Prediction Time'
        Label:
            text: '10s'
        Label:
            text: 'Temperature Reading'
        Label:
            text: str(round(root.temp,1))
        Label:
            text: 'Predicted Temperature'
        Label:
            text: str(root.pred_temp) 
''')

class MainWidget(GridLayout):
    
    timepassed = NumericProperty()
    temp = NumericProperty()
    pred_temp = NumericProperty()
    
    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        Clock.schedule_interval(self.increment_temp, .1)
        self.increment_temp(0)
                                                                                #save the machine learning model
    def increment_time(self, interval):                             
        self.timepassed = time.time() - self.st
          
    def increment_temp(self, interval):
        self.temp = db.child('r_pi').child("temp").get().val()
        if db.child('r_pi').child("temp_list").get().val() == 'done':
            temps_at_times = []
            for i in range(0,10):
                current_temp = db.child('temps_at_times').child(str(i)).get().val()
                temps_at_times.append(current_temp)
            db.child('r_pi').update({"temp_list":"waiting"})
            temps_at_times = np.array([temps_at_times]).astype(float)
            predicted_final_temp = model.predict(temps_at_times)
            print(type(float(predicted_final_temp[0])))
            self.pred_temp = float(predicted_final_temp[0])
            
    def start_time(self):
        self.st = time.time()
        Clock.schedule_interval(self.increment_time, .1)
        self.increment_time(0)
        
    def begin_twodee(self):
        db.child("starter").update({"start_twodee": "True"})
        
               
class Timer(App):
    def build(self):
        return MainWidget()

#WHOLE SCRIPT
        
db.child("starter").update({"start_simple_temp": "True"})
x_data,y_data = preprocess(data)
results,model = train_model(data,0.1)
Timer().run()
db.child("starter").update({"start_simple_temp": "False"})
db.child("starter").update({"start_twodee": "False"})
