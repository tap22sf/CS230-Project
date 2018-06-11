# Prediction of postoperative opioid continuation in patients undergoing lumber fusion (Stanford CS230 Project)

Given the addictive properties of opioid medications, they are not recommended after the immediate recovery period following surgery. However, approximately 20\% of patients undergoing lumbar fusion meet criteria for “chronic opioid use” more than 3 months after surgery [1]. Despite growing awareness about the opioid epidemic sweeping the US, we do not have currently any way to predict which patients will continue taking opioids for prolonged periods of time following surgery. In this project, we attempted to predict which patients were at risk for continued opioid usage after 3 months following lumbar fusion using administrative healthcare data. After conversion of diagnosis codes into their corresponding 300-dimensional encodings, an alternative training dataset was also created using principal components analysis (PCA) to reduce the dimensionality of these encodings into their most informative two dimensions. Ultimately, our model was able to achieve similar predictive accuracy on these two datasets, with AUCs of 0.725 and 0.728 for the raw encodings and dimensionality-reduced datasets, respective. This network performed as well or better than logistic regression (AUC = 0.721) and random forest (AUC = 0.724) classifiers trained on the same dataset. 

## Helpful Links for getting started

[Tensorflow Installation](https://www.tensorflow.org/install/)

## Background
Despite increasing publicity of the current opioid epidemic, opioids are currently the most commonly prescribed medication for low back pain [1]. Although surgery can be an option for patients with particularly severe structural disease, we  currently do not have any way to predict which patients will continue opioid usage following surgery and which will be able to stop. With the increasing popularity of deep learning and the availability of larger databases in the past few years, this sort of prediction has become increasingly feasible. If we can better predict who is most likely to continue opioid usage following lumbar fusion, we can better divert counselling to these patients and help them stop taking opioid medications, decreasing their risk for addiction and overdose.  

## Database
This project utilizes data from inpatient, outpatient, and pharmacy settings from the MarketScan Commercial Claims and Encounters Database and Medicare Supplemental and Coordination of Benefits Database. These data encompass health care claims submitted on behalf of individuals enrolled in private insurance plans and Medicare through a participating employer, health plan, or government organization. Both inpatient and outpatient data (including information on diagnosis, date of service, demographics, and employer information, among others) were queried to select our cohort of patients undergoing lumbar fusion. To obtain information on prescription drug use, we used the associated drug prescription database, which includes information on all prescriptions covered by insurance that were filled by the patient, along with dosage, drug identification number, day supply, and prescription date. The data are a common source of data for analyses of health care utilization and spending [2, 3, 4, 5].

As the database is not available for public use, we created a sample dataset (Folder Sample Dataset) consisting of 10,000 patients. The features for each patient were generated such that the distribution of that feature is the same as the original dataset.

## Running

The weights of the optimal model have been included in the folder "Weights". The characteristics of the optimal model are the following:
+ n = 500 (Number of nodes of first layer. Consecutive layers have 2*n nodes)
+ l = 3 (Number of layers of network)
+ lr = -3.0 (Exponent of learning rate, i.e. 1e-3)
+ epochs = 100
+ dropout = 0.5
+ l2reg = 0.0 (L2 regularization)
+ bz = 8192 (batch size)

To create virtual environment:  
`virtualenv -p python3 .env`  
`source .env/bin/activate`  
`pip install -r CS230-Project/requirements.txt`

To generate predictiviness metrics for a given model:  
Run Opioids.py

If the model need to be trained:  
Run Opioids.py having loadPrevModel = False

If the model weights are loaded from a previous training:  
Run Opioids.py having loadPrevModel = True

## Results: 

The following graph shows the AUC results for the optimal model which was trained using the information of 126,763 patients:  
![](https://github.com/tap22sf/CS230-Project/blob/master/Images/Results.jpg)  

## Citations
(1) Mark TL, Vandivort-Warren R, Miller K: Mental health spending by private insurance: implications for the mental health parity and addiction equity act. Psychiatr Serv 63:313–318, 2012.  
(2) Stephens JR, Steiner MJ, DeJong N, Rodean J, Hall M, Richardson T, et al: Healthcare utilization and spending for constipation in children with versus without complex chronic conditions. J Pediatr Gastroenterol Nutr 64:31–36, 2017.  
(3) Veeravagu A, Cole TS, Jiang B, Ratliff JK, Gidwani RA: The use of bone morphogenetic protein in thoracolumbar spine procedures: analysis of the MarketScan longitudinal database. Spine J 14:2929–2937, 2014.  
(4) Wu J, Thammakhoune J, Dai W, Koren A, Tcherny-Lessenot S, Wu C, et al: Assessment of dronedarone utilization using US claims databases. Clin Ther 36:264–272, 272.e1–272.e2, 2014.  

# License
MIT License

Copyright (c) 2018 Felipe Kettlun, Chloe O'Connell, Thomas Petersen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Resources
[paper](http://arxiv.org/pdf/1508.06576v2.pdf)  
[encodings of medical information](http://people.csail.mit.edu/dsontag/papers/ChoiChiuSontag_AMIA_CRI16.pdf)  (will be used for ICD-9 code processing)  
[link to download encodings of ICD-9 codes](https://github.com/clinicalml/embeddings/blob/master/claims_codes_hs_300.txt.gz)  
[link to CDC MME conversion table](https://www.cdc.gov/drugoverdose/resources/data.html)  
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS  
[adam]: http://arxiv.org/abs/1412.6980  
[license]: LICENSE.txt
