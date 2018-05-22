import numpy as np
from datetime import datetime
import csv
import json

CSV_FILE = "data_1m.csv"
JSON_FILE = "nn_1m.json"

class Data:
    def __init__(self):
        self.data = []
        self.nn_in_data = []
        self.nn_out_data = []
        self.json_data = []

    def get_data(self, original, new):
        """ 
        Fetch the exchange data from the API, and format and save 
        into self.data list, ready to be formatted to JSON
        """
        with open(CSV_FILE) as f:
            reader = csv.reader(f, delimiter="\n")
            for row in reader:
                row_pre = row[0].split(",")
                try:
                    self.data.append([row_pre[0],float(row_pre[1]),float(row_pre[2]),float(row_pre[3]),float(row_pre[4]),float(row_pre[5])])
                except ValueError:
                    continue
        del(self.data[0]) ## removes header
        
    def convert_to_nn_input(self, length=20, output_offset=1):
        ## Start structuring NN inputs, and skip those which fall into forex close times
        for input_index in range(len(self.data)):
            invalid_flag = False
 
            ## Add elements for specified length
            nn_input = []
            nn_output = []
            for input_pointer in range(length):
                ## Check if current pointer exists:
                try:
                    ## Check if current pointer is for price during close
                    if(self.data[input_index+input_pointer][5] == 0):
                        invalid_flag = True
                        break
                    nn_input.append(self.data[input_index+input_pointer][1:5])
                except IndexError:
                    invalid_flag = True
                    break
            ## Add result element for each input
            try:
                nn_output = self.data[input_index+(length-1)+output_offset][1:5]
                ## TODO fix output if during close
            except IndexError:
                continue
            ## If any elements out of range or during closed hours, ignore
            if invalid_flag == True:
                continue
            self.nn_in_data.append(nn_input)
            self.nn_out_data.append(nn_output)
             
    def save_to_json(self):
        """
        Convert the sets of inputs and outputs into JSON, ready for nn_model.py
        for training
        """
        for i in range(len(self.nn_in_data)):
            self.json_data.append([self.nn_in_data[i], self.nn_out_data[i]])
        with open(JSON_FILE, "w") as f:
            json.dump(self.json_data, f, indent=4)
                    

    def stringToDate(self, string):
        ## Take format: "dd.MM.yyyy hh:mm::ss.ssss"
        date = datetime.strptime(string, "%d.%m.%Y %H:%M:%S.000")
        return date
            

if __name__ == "__main__":
    d = Data()
    d.get_data("EUR","USD")
    d.convert_to_nn_input(5)
    d.save_to_json()

