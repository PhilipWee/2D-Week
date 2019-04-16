import os
import glob
import time
import numpy as np

os.system('modprobe w1-gpio')  #registers the sensor connected to GPIO4 using 1-Wire system
os.system('modprobe w1-therm') #add temp measuring ability to using 1-Wire system

#find the file (w1_slave) where the readings are being recorded
base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'


def read_temp_raw():
    f = open(device_file, 'r') #open file where temp is recorded
    lines = f.readlines()      #read the temp in its original raw ugly form
    f.close()                  #close the file
    return lines               #return raw data

def read_temp():
    lines = read_temp_raw()                   #read raw values form earlier function
    while lines[0].strip()[-3:] != 'YES':     #filters out the bad readings
        time.sleep(0.2)                       
        lines = read_temp_raw()               #read the raw data from the sensor
    equals_pos = lines[1].find('t=')          
    if equals_pos != -1:                      #returns the 
        temp_string = lines[1][equals_pos+2:] #locates the temp part of the raw data 
        temp_c = float(temp_string) / 1000.0  #converts it into celsius
        return temp_c                         #returns the temp in celsius

def measure_for_increase():
    print("Initiating measurement sequence...")   
    while True:             
        T = read_temp()                                      #create a variable T for current temp
        print(T)                                             #print out the current temp continuously
        time.sleep(0.001)                                    #every 0.001s
        if T > 24:                                           #when the temperature rises above 24,
            print('initial temp:{}'.format(T))               #print out this temperature               
            nested_list = [[],[],[],[],[],[],[],[],[],[]]    #created an empty nested list of 10 lists
            
            for i in range(10):                              #for values of i from 0 to 10
                time.sleep(1)                                #wait for 1sec
                inner_list = nested_list[i]                  #call out the ith list in the nested list
                inner_list.append(i+1)                       #add the time from initial reading that the measurement is taken
                inner_list.append(read_temp())               #add the measurement
                print(inner_list)                            #print the data list for t = i+
                holy_array = np.array(nested_list)
            
            return(holy_array)                               #return this array
                
def measure_for_decrease():                                  #same as previous function,
    print("Initiating measurement sequence...")              #but for decreasing temperature
    while True:
        T = read_temp()
        print(T)
        time.sleep(0.001)
        if T < 24:
            print('initial temp:{}'.format(T))
            nested_list = [[],[],[],[],[],[],[],[],[],[]]
            
            for i in range(10):
                time.sleep(1)
                inner_list = nested_list[i]
                inner_list.append(i+1)
                inner_list.append(read_temp())
                print(inner_list)
                holy_array = np.array(nested_list)
            
            return(holy_array)

def get_training_data():
    print("Checking temp change type (Increase/Decrease)... ...")
    print("Please be patient... ...")
    while True:
          t1 = read_temp()            #reads a temp, then reads again after 0.001secs
          print(t1)                   
          time.sleep(0.001)
          t2 = read_temp()
          print(t2)
          if t2 > t1:                                  #if the change is (+)ve,then 
              print("Temp is increasing...")           #print "increasing"
              training_data = measure_for_increase()   #run the measure function for increasing temp
              return training_data                     #and return the measurements
          if t2 < t1:                                  #if the change is (-)ve, then
              print("Temp is decreasing...")           #print "decreasing"
              training_data = measure_for_decrease()   #run the measure function for decreasing temp
              return training_data                     #and return the measurements
          if t2 == t1:                                 #keep checking the temp if the temp does not change
              pass
            

def gen_text_data(data):                               #generate text data from the measurements
    text = ""
    for inner_list in data:
        text += str(inner_list[0]) + ' ' + str(inner_list[1]) + '\n' 
    return text                                        #transferring and organisation of the data

def measure_final_temp():                                #create an input for user to put in the externally measure temp
    for i in range(11):
        time.sleep(10)
        print(str(10+i*10) + 'seconds passed. Temp: ' + str(read_temp()))
        
    text = 'Final Temp: ' + str(read_temp()) +'\n'
    return text

def full_text():
    data = get_training_data()        #run the experiment and get the measurements
    text_data = gen_text_data(data)   #convert the measurements into text data
    text = "Experiment: \n"           #insert a title for the data
    text += measure_final_temp()        #insert the user input of the actual final temp
    text += text_data                 #insert the converted-to-text measurement data
    print(text)                       #print all this
    return text

#print(measure_final_temp())

f = open("PENIS.txt", "a")    #create a new file to place the data
f.write(full_text())          #run the entire procedure that gives u the text data and input in the text
f.close()                     #save and close the file