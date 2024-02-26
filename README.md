<h1 align="center"> Supervised Regression-Yes Bank Stock Closing Price Prediction </h1>

![--------------------------------------------------------------------------------------------](https://github.com/andreasbm/readme/blob/master/assets/lines/grass.png)


![Yes_Bank_SVG_Logo svg](https://github.com/sanjaypatil91/Regression-Yes-Bank-Stock-Closing-Price-Prediction/assets/102731353/e2ad4fc4-5634-4de5-b35c-ed048e9b8a21)


<p align="center"> 
<img src="GIF/google play.gif" alt="Animated gif" height="282px">
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
## Table Of Contents

- [Project Summary](project-Summary)
- [GitHub Link](GitHu-blink)
- [Problem Statement](problem-statements)
- [Project Files Description](project-files-description)
- [Dataset Contents](dataset-contents)
- [Variables Details Of Dataset](variables-details-of-dataset)
- [Technologies Used](technologies-used)
- [Steps Involved](#steps-involved)
- [Key Insights](key-insights)
- [Conclusion](conclusion)

## 📋 Project Summary
TThe "Regression-Yes Bank Stock Closing Price Prediction" capstone project aims to develop an accurate and reliable model for forecasting the closing prices of Yes Bank stocks. Yes Bank, a prominent financial institution, is subject to various market dynamics and economic factors that influence its stock prices. The project utilizes regression analysis, a powerful statistical method, to create a predictive model that can assist investors, traders, and financial analysts in making informed decisions.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


##  💾 Project Files Description

<p>This Project includes 1 google colab notebook and 1 data set in the fromat of csv file:</p>

### Executable Files:
- [Regression-Yes_Bank_Stock_Closing_Price_Prediction_ipunb
](https://github.com/sanjaypatil91/Regression-Yes-Bank-Stock-Closing-Price-Prediction) - Includes all functions required for clustering operations.

### Output:
- [Google Colab](https://github.com/San13deep/Play-Store-App-Review-Analysis/blob/main/Play_Store_App_Review_Analysis_Capstone_Project.ipynb) - All the outputs are visible in the provided colab notebook.

### Input Files:
  <li><b>data_YesBankPrices.csv</b> - It contains the basic details of the data like opening & closing price, highrnand low price and date etc.</li>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## Dataset Contents

- This **[data set](https://drive.google.com/file/d/19fVBgoQJr5jkupe0O9Ht8VeMqSVaBJae/view?usp=share_link)** contains one csv  files of data and there are five variables in this dataset. i.e. 'Date', 'Open', 'High', 'Low', 'Close'. Out of these..
- Key variables in the dataset includes :

**Apps Dataset:**

| variables | Details |
| --------------------- | ---------------------- |
| Independent Variable are:| Date, Open, High and Low |
| Dependent Variable: Close. | This section gives the closing price of the stock|
| Date | Represents the date of the stock market data. |
| Open | Denotes the opening price of the stock on a particular date. |
| High | Indicates the highest price reached by the stock during the trading day. |
| Low  | Represents the lowest price reached by the stock during the trading day. |
| Close| Denotes the closing price of the stock on a particular date.|

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


## 📋Problem Statements
In the complex landscape of financial markets, predicting stock prices accurately is a challenging task. Investors, traders, and financial analysts face difficulties in making well-informed decisions due to the lots of factors influencing stock prices. The specific challenge addressed in this capstone project is the need for a reliable regression-based model to predict the closing prices of Yes Bank stocks. The goal is to develop a robust predictive model that leverages historical stock data, financial indicators, and relevant macroeconomic variables to forecast Yes Bank's stock closing prices.

********************************************************************************************************************************************************************

## 📔 **What are the technolgies used?**
1. Python
2. Numpy library
3. Pandas library
4. Plotly
5. sklearn ML Module
6. Scipy
7. Google Colab Notebook
8. Linear Algebra
9. Matplotlib
10. Seaborn
   
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 📖	Steps Involved

### Step 1: Data Collection
Collect the historical stock prices of Yes Bank from January 2005 to September 2020. Include variables such as opening price, highest price, lowest price, and closing price and date

### Step 2: Data Cleaning, Preprocessing and Feature Engineering
Clean the dataset by removing any missing values, outliers, or errors. Preprocess the dataset by scaling or normalizing the features as necessary.Checking for and Dealing with multicollinearity present in our dataset.Applying the log transform to deal with positively skewed data.Scaling the data and splitting it into train and test sets.

### Exploratory Data Analysis
After establishing a good sense of each feature, we proceeded with plotting a pairwise plot between all the quantitative variables to look for any evident patterns or relationships between the features. There is a high variance in the number of installs and in number of reviews. To overcome this problem, we add two new columns to the data frame named: log_installs and log_review, which contain the logarithmic values of installs and review columns, respectively.

### Single, Biavriate and Mulitvariate Analysis
After that we analysis all the columns one by one to examine whether the particular column contain some useful information or not:

### Step 3: Model Implementation :-
Divide the dataset into training and testing sets. Implement Multiple Linear Regression ,Lasso Regression, Ridge Regression, Elastic net Regression, knn regressor and Random forest regressor models to predict the closing price of Yes Bank stock. Tune the models by adjusting the hyperparameters to optimize performance.
Fitting various models on our data and optimizing them via cross-validation.
Using these models to make predictions on test and train data.
The Models implemented are :-
1. Linear Regression
2. Lasso Regression
3. Ridge Regression
4. Elastic Net Regression

### Step 4: Model Evaluation
Evaluate the models' performance using root mean squared error (RMSE) and mean absolute error (MAE) metrics. Compare the performance of the models to determine which one performs better.

### Step 5: Data Visualization :-
Using several kinds of charts like Line chart, scatter plot, heatmap, pair plot, distplot, boxplot etc 
to better visualize data and understand correlation and trends.

### Step 6: Model performance comparison :-
Comparison of all implemented models using various Regression evaluation metrics like Mean 
absolute error, Mean squared error, RMSE, R-squared and Adjusted R-squared.

###  Step 5: Results and Conclusion
Present the results of the regression analysis. Discuss the implications of the results and their potential impact on investors. Provide conclusions and suggestions for future research.Drawing some insights from the data and the predictions made by our various predictive models on unseen (test) data.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 📋 Conclusion & Key Insights :

1. There is a high correlation between the dependent and independent variables. This is a good 
thing as we can make really accurate predictions using simple linear models.
2. We implemented several models on our dataset in order to be able to predict the closing price
and found that Elastic Net regressor is the best performing model with Adjusted R2 score
value of 0.9932 and it scores well on all evaluation metrics.
3. All of the models performed quite well on our data giving us the accuracy of over 99%..
4. We found that there is a rather high correlation between our independent variables. This 
multicollinearity however is unavoidable here as the dataset is very small.
5. We found that the distribution of all our variables is positively skewed. so we performed 
log transformation on them.
6. Using data visualization on our target variable, we can clearly see the impact of 2018 fraud 
case involving Rana Kapoor as the stock prices decline dramatically during that period.
7. With our model making predictions with such high accuracy even on unseen test data, 
we can confidently deploy this model for further predictive tasks using future real data.


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
