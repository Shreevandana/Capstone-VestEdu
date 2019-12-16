# Interest Rate Prediction

Python file to predict the interest rate distribution for DOE based on debt and income from Lending Club.

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

The DOE xlsx *("NewDOEData_corrected.xlsx")* and Lending Club csv *("LC_LoanDataQ2_19.csv")* should be in the same directoy as this script.
To run the script on your terminal type -

```bash
python3 Interest_Rate_Prediction.py 
```

## Sample Output 

Note: These Values are subject to change on each iteration.


The mean of the predicted distribution is 12.043911

The maximum value of the predicted distribution is 13.705189

The minimum value of the predicted distribution is 10.420234

The standard deviation of the predicted distribution is 0.71117604

The median of the predicted distribution is 12.58776

A new excel filed called "NewDOEData_corrected.xlsx" will be written in the same directory. This file will have DOE data along with predicted interest rate.


## Reference
This script was  created as a part of the project in the Capstone class (DSC 483, http://www.sas.rochester.edu/dsc/news/2018/2018-0613-capstones.html) at the University of Rochester, New York.
