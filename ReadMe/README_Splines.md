# Feature Importance using Loan Status

Python file to give importance of attributes like rank of university, level of education and field of study based on loan status classification

## Installation
This file uses Python3.

Use the package manager [pip] to install all necessary dependencies 

```bash
pip3 install pandas
```
```bash
pip3 install numpy
```
```bash
pip3 install xgboost
```
```bash
pip3 install sklearn
```
```bash
pip3 install xlrd
```
```bash
pip3 install pyearth
```

## Usage

The DOE xlsx *("alternative_cdr_long_format.csv")* should be in the same directoy as this script.
To run the script on your terminal type -

```bash
python3 Feature_Imp_Loan_Status.py 
```

## Sample Output

Note: These Values are subject to change on iteration. 

The importance of forbes_university_rank in CDR 2 model is 7.884675689181249

The importance of forbes_university_rank in CDR 3 model is 14.85087413347333

The importance of major_Business in CDR 2 model is 2.9723705104266194

The importance of major_Business in CDR 3 model is 3.509726566393665

The importance of major_STEM in CDR 2 model is 0.4400918655485255

The importance of major_STEM in CDR 3 model is 0.08827684029589886

The importance of major_Arts in CDR 2 model is 0.0

The importance of major_Arts in CDR 3 model is 0.0

The importance of degree in CDR 2 model is 85.44239435939981

The importance of degree in CDR 3 model is 81.22954440423878

The MSE for the CDR2 model is 0.7801559307692119

The MSE for the CDR3 model is 0.8234943914725866
## Reference
This script was  created as a part of the project in the Capstone class (DSC 483, http://www.sas.rochester.edu/dsc/news/2018/2018-0613-capstones.html) at the University of Rochester, New York.
