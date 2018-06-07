import csv
from dataParser import *
import os

absFilePath = os.path.abspath(__file__)
fileDir = os.path.dirname(absFilePath)
parentDir = os.path.dirname(os.path.dirname(os.path.dirname(fileDir)))
dataFile1Path = os.path.join(parentDir, 'Medical Database/scaled_PCs.csv')
#dataFile1Path = os.path.join(parentDir, 'Medical Database/withEncodings.csv')

# Read the datasets
Xpath= dataFile1Path

with open(Xpath, "r") as f:
    reader = csv.reader(f)
    stringNames = next(reader)

xusecols = (stringNames)

print(xusecols)
            
yusecols = (
    #"num",
    #"subj",
    #"totalPostopSupply",
    #"totalPostopDays",
    "noOpiates",
    #"comp",
    #"cost_1yr",
    #"cost_2yr",
    #"readmit",
    #"length_stay",
    #"inpt_cost",
    #"cost_90d",
    #"dis_home"
)

