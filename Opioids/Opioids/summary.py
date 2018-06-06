import os
import csv
import numpy as np

#
# Read the results directory
#
resultsDir = r'Results\BatchSize'
resultsDir = r'Results\Model Arch Search'
resultsDir = r'.'

# Open a summary file
with open(resultsDir+'\\summary.csv', 'w', newline='') as summaryfile:
    writer = csv.writer(summaryfile, delimiter=',')
    writer.writerow(["learning_rate", "dropout_rate", "batch_size", "nodes", "layers", "trainingLoss", "devLoss", "testLoss", "trainingAccuracy",  "devAccuracy", "testAccuracy"])

    for file in os.listdir(resultsDir):

        fields = os.path.splitext(file)[0].split('+')
        if fields[0] == "metrics":
            metric = np.loadtxt(resultsDir+'\\'+file, delimiter=',', skiprows=1)
            print (file)

            writer.writerow ([
                fields[2],
                fields[4], 
                fields[6], 
                fields[8], 
                fields[10], 

                metric[0][0],
                metric[0][1],
                metric[0][2],
                metric[1][0],
                metric[1][1],
                metric[1][2]])
