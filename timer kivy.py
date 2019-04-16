from kivy.app import App
from kivy.lang import Builder
from kivy.properties import NumericProperty,StringProperty
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from libdw import pyrebase
import pickle
import time

#train the machine learning model
#Necessary Imports
import numpy as np
from sklearn import linear_model 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

filename = 'NUDES'

# code to read txt file

# open the file
f = open('{}.txt'.format(filename),'r')

#create an array to store the data
data = [['final temp','1s','2s','3s','4s','5s','6s','7s','8s','9s','10s']]

# iterate through the lines in the file
for line in f:
    if line.strip() == 'Experiment:':
        data_row = []
        
        #Get the final temp
        final_temp_line = f.readline()
        final_temp_array = final_temp_line.split(" ")
        final_temp = final_temp_array[2].strip()
        data_row.append(final_temp)
        
        
        #Get the individual temp
        for i in range(10):
            temp_line = f.readline()
            temp_array = temp_line.split(" ")
            temp_at_time = temp_array[1].strip()
            data_row.append(temp_at_time)
        
        data.append(data_row)

#print(np.array(data))
f.close()

def preprocess(data):
    #save the dataset
    dataset = data
    #split into x and y
    x_data = []
    y_data = []
    for row_no in range(1,len(dataset)):
        print(dataset[row_no][0])
        y_data.append(dataset[row_no][0])
        x_data_row = []
        for col_no in range(1,len(dataset[0])):
            x_data_row.append(dataset[row_no][col_no])
        x_data.append(x_data_row)
#     x_data = dataset[1:,1:]
#     y_data = dataset[1:,0]
    return x_data,y_data

x_data,y_data = preprocess(data)
#print(np.array(x_data),y_data)

# write a code to split the data to train and test
def prepare_train_test(data,percentage_for_test):
    x_data,y_data = preprocess(data)
    #split further into testdata and practice data
    x_train,x_test,y_train,y_test = train_test_split(x_data,\
                                                     y_data,\
                                                     test_size = percentage_for_test)
    x_train =  np.array(x_train).astype(float)
    x_test =  np.array(x_test).astype(float)
    y_train =  np.array(y_train).astype(float)
    y_test =  np.array(y_test).astype(float) 
    
    return x_train,x_test,y_train,y_test

#print(prepare_train_test(data,0.5))
    
# write a code to train the model
# the function should return the trained model
def train_model(data, percentage_for_test):
    
    x_train,x_test,y_train,y_test = prepare_train_test(data,percentage_for_test)
    
    #create the linear regression model
    model = linear_model.LinearRegression()
    model.fit(x_train,y_train)
    #predict the y values for a given c
    #print(x_test)
    y_pred = model.predict(x_test)
#     print(y_pred)
    
    #calculate the R2,MSE,coef and intercept
    MSE = mean_squared_error(y_test, y_pred)
    R2 = r2_score(y_test,y_pred)
    coef = model.coef_
    intercept = model.intercept_
    #generate the results dictionary
    
    
    abs_error_vector = np.abs(y_test - y_pred)
    #print('the abs error is ' + str(abs_error_vector))
    
    results = {"mean squared error" : MSE,
               "intercept" : intercept,
               "coefficients" : coef,
               "r2 score" : R2}
    return results,model

results,model = train_model(data,0.1)
#print(results)

url = 'https://twod-dbdd4.firebaseio.com/'
apikey = 'AIzaSyDyw9NUmAGv7alG-bVGzCClfb88MNeqBnc'
config = {"apiKey": apikey,"databaseURL": url,}
firebase = pyrebase.initialize_app(config)
db = firebase.database()
var = db.child('r_pi').child("temp").get()
 
Builder.load_string('''
<MainWidget>:
    GridLayout:

        cols: 4
        width:root.width
        height:root.height

        Label:
            text: 'Current Time' 
        Button:
            text: str(root.timepassed)
            on_press: root.start_time()
        Label: 
            text: 'Prediction Time'
        Label:
            text: '10s'
        Label:
            text: 'Temperature Reading'
        Button:
            text: str(round(root.temp,1))
            on_press: root.start_temp()
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
        
    def start_temp(self):
        Clock.schedule_interval(self.increment_temp, .1)
        self.increment_temp(0)
               
class Timer(App):
    def build(self):
        return MainWidget()

Timer().run()

