# CS230 Project

Placeholder for a project description

## Helpful Links for getting started

[Markdown Cheatsheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)

[Example from TA's on Friday - Neural Style](https://github.com/anishathalye/neural-style)

[Tensorflow Installation](https://www.tensorflow.org/install/)

# Data source
This project utilizes data from inpatient, outpatient, and pharmacy settings from the MarketScan Commercial Claims and Encounters Database and Medicare Supplemental and Coordination of Benefits Database. These data encompass health care claims submitted on behalf of individuals enrolled in private insurance plans and Medicare through a participating employer, health plan, or government
organization. Both inpatient and outpatient data (including information on diagnosis, date of service, demographics, and employer
information, among others) were queried to select our cohort of patients undergoing lumbar fusion. To obtain information on prescription drug use, we used the associated drug prescription database, which includes information on all prescriptions covered by insurance that were filled by the patient, along with dosage, drug identification number, day supply, and prescription date. The data are a common source of data for analyses of health care utilization and spending (1, 2, 3).  

## Running

To train the newtwork:

`python train_network --dataset train_dataset`

To evaluate the test dataset: 

`python evaluate_dataset --dataset test_dataset --output predictions`


## Inline Image example: 
![alt text](images/Original.jpg|width=200)

![alt text](images/Style.jpg) 

![alt text](images/Style Transfer.jpg "Example image transfer")



## Citation

# License
Copyright (c) 2015-2018 Anish Athalye. Released under GPLv3. See
[LICENSE.txt][license] for details.

[net]: http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
[paper]: http://arxiv.org/pdf/1508.06576v2.pdf
[l-bfgs]: https://en.wikipedia.org/wiki/Limited-memory_BFGS
[adam]: http://arxiv.org/abs/1412.6980
[ad]: https://en.wikipedia.org/wiki/Automatic_differentiation
[lengstrom-fast-style-transfer]: https://github.com/lengstrom/fast-style-transfer
[fast-neural-style]: https://arxiv.org/pdf/1603.08155v1.pdf
[license]: LICENSE.txt
