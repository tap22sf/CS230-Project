# CS230 Project

Placeholder for a project description

# Data source
This project utilizes data from inpatient, outpatient, and pharmacy settings from the MarketScan Commercial Claims and Encounters Database and Medicare Supplemental and Coordination of Benefits Database. These data encompass health care claims submitted on behalf of individuals enrolled in private insurance plans and Medicare through a participating employer, health plan, or government
organization. Both inpatient and outpatient data (including information on diagnosis, date of service, demographics, and employer
information, among others) were queried to select our cohort of patients undergoing lumbar fusion. To obtain information on prescription drug use, we used the associated drug prescription database, which includes information on all prescriptions covered by insurance that were filled by the patient, along with dosage, drug identification number, day supply, and prescription date. The data are a common source of data for analyses of health care utilization and spending (1, 2, 3, 4).  

## Related Projects

## Running

To train the newtwork:

`python train_network --dataset train_dataset`

To evaluate the test dataset: 

`python evaluate_dataset --dataset test_dataset --output predictions`





## Citation

If you use this implementation in your work, please cite the following:

```
@misc{athalye2015neuralstyle,
  author = {Anish Athalye},
  title = {Neural Style},
  year = {2015},
  howpublished = {\url{https://github.com/anishathalye/neural-style}},
  note = {commit xxxxxxx}
}
```

## License

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

## Citations
Mark TL, Vandivort-Warren R, Miller K: Mental health spending by private insurance: implications for the mental health parity and addiction equity act. Psychiatr Serv 63:313–318, 2012.  
