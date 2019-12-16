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

## Usage

The DOE xlsx *("NewDOEData_corrected.xlsx")* should be in the same directoy as this script.
To run the script on your terminal type -

```bash
python3 Feature_Imp_Loan_Status.py 
```

## Sample Output

Note: These Values are subject to change on iteration. 

Mean Value of Train AUC over ten runs 0.9207184105471681 

Mean Value of Test AUC over ten runs 0.924514282377972 

Mean Value of Train Accuracy over ten runs 93.79526977087951 % 

Mean Value of Test Accuracy over ten runs 94.22025129342202 % 

The feature importance for Level is 98.34814667701721 % 

The feature importance for Rank is 1.0098932310938835 % 

The feature importance for Field is 0.6419565994292498 % 

## Reference
This script was  created as a part of the project in the Capstone class (DSC 483, http://www.sas.rochester.edu/dsc/news/2018/2018-0613-capstones.html) at the University of Rochester, New York.
