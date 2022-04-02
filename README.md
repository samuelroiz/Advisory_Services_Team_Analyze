# Advisory_Services_Team_Analyze

<p>
Created a new cryptocurrency investment portfolio for the Advisory Services Team of a financial consultancy. Clients are interested in offering a new cryptocurrency investment portfolio for their customers. The company, however, is lost in the vast universe of cryptocurrencies. The company asked to create a report that includes what cryptocurrencies are on the trading market and determines whether they can be grouped to create a classification system for this new investment. </p>
<p> The organization gave raw data to process it to fit the machine learning models. Using unsupervised learning and several clustering algorithms to explore the cryptocurrencies can be grouped with other similar cryptocurrencies. Create data visualization to share the findings with the investment bank and communicate with the non-data science audience. 
</p>

## Data Clean Steps

<p>
Advisory Services Team of a financial consultancy gave and created an unfinished data frame. The data frame has seven columns with one thousand two hundred fifty-two rows. The seven columns are named Unnamed: 0,	CoinName,	Algorithm,	IsTrading,	ProofType,	TotalCoinsMined, and	TotalCoinSupply.
</p>

![Given Dataframe](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/given_data_frame_from_crypto_company.png)

<p>
 Data frames should not contain any null values or it will create conflicts with machine learning and plotting. A clean data frame is a data frame that does not include any null values. The given data frame has 508 null values in column TotalCoinsMined and the only column to contain null values. 
 </p>
 
 ![Check null val. in data frame](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/check_null_val_and_duplicates.png)

<p>
There are two options to approach this route. You can drop the TotalCoinsMined column since it is the only column that contains null values. The other option is drop <b> ALL </b> null values in the data frame. In some cases is not to drop the null values since they can be valuable information. In this case, they are not valuable. 
</p>

![Null val. dropped](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/dropped_null_values.png)

<p>
It is important to double-check if the code went through. In this case, the code checks if the data frame has any null values. The microcode shows the following data frame does not have nulls then the argument worked. 
</p>

![Check null val. in data frame again](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/check_null_val_and_duplicates_again.png)

<p>
The data frame column isTrading has two boolean values. If a row is true then cryptocurrency is still trading. Else it is not actively trading. Due to the confusion of boolean values, the column will convert from boolean to integer values to avoid conflicts. One will be actively trading, zero will be no trading. After converting the column, the data frame is filtered to rows containing one. The company only wants cryptocurrencies that are currently trading. 
</p>

![Filter data frame of Crypto Currencies still trading](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/filter_dataframe_to_isTrading_1_only_aka_true_only.png)

<p>
The data frame has a column named TotalCoinsMined and has some empty values or zero. The company only wants cryptocurrencies that have coins mined more than zero. Total coins mined are essential to avoid scams and reduce the risk of counterfeiting, duplicates, or repeating the same coin more than once. In this case, the data frame will filter out cryptocurrencies rows that have not been mined. 
</p>

![Filter data frame of Crypto Mined greater than zero](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/filter_dataframe_to_crypto_mined_greater_than_0.png)

<p>
The old data frame had seven columns with one thousand two hundred fifty-two rows. The clean data frame has four columns with five hundred thirty-two rows and active trading currencies. Column Unnamed: 0, IsTrading, and CoinName are dropped for machine learning. 
</p>

![Clean Data frame outcome](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/final_clean_data_outcome_from_crypto_company.png)

## Machine Learning Steps


### Get_dummies() Section

#### Example.

<p>
Here is an example of how pd.get_dummies() work.
</p> 

![Get_Dummies() Part 1](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/example_get_dummies_part_1.png) <p> The following data frame is named <i> preview_get_dummies </i> displays two non-numeric columns. </p>

<p>
  Then apply the get_dummies() code: <b> pd.get_dummies(<i> preview_get_dummies </i>) </b>
</p>

![Get_Dummies() Part 2](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/example_get_dummies_part_2.png)

<p>
Now the <i> preview_get_dummies </i> data frame has numeric columns which means it meets the standards of machine learning models.
</p>

### Get Dummies Outcome
<p>
Apply the get_dummies() code to the data frame and the data frame ends up having a shape of five hundred thirty-two and ninety-eight columns. 
</p>

![Get Dummies Outcome](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/get_dummies_outcome.png)

### Standard Scaler Section

#### How does the Standard scaler work and what is it doing to the data? 

<p> Standard Scaler has a formula, let's assume the formula is <b> z = (x - u) / s </b> where <b> z </b> is the scaled data, <b> x </b> is to be the scaled data, <b> u </b> is the average of the training samples, and <b> s </b> is the standard deviation of the training samples. The average or mean is the sum of all data points divided by the number of data points. In this case, <b> u </b> is going to add up all of the training samples divided by several training samples. The standard deviation formula starts by taking the given values and placing them in one column. Square each value and place them in a second column. Find the sum of all values in the first column and square it. The value is divided by the number of data points in first the column and called this number <b> i </b>. Find the sum of all values in the created second column. Once find the value, subtract it by <b> i </b>. The outcome will be divided by the number of data points minus one. Value leads to the <b> variance </b> of the sample and data. Finally, the variance will be square rooted leading to the value standard deviation of the data. 
</p>

#### Example of Standard Scaler
<p>
To demonstrate the algorithm and how it functions, consider the data set {1,2,3,4,5}. The data set consists of 5 one dimensional data points and each data point has one feature. Now apply the standard scaler() to the data. The data set becomes {−1.41,−0.71,0.,0.71,1.41}.
</p>

<p>
  The following example takes all of the data points and converts them to a closer range of 0 to 1. Standard scaler helps prevent outliers and keep the data closer to each other rather than gaps. 
  </p>

### Standard Scaler Outcome 

<p>
 Apply the standard scaler code to the dummies' data frame to demonstrate the algorithm and drive the data closer to each other. 
</p>

![Standard Scaler X Values](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/standard_scaler_x_val.png)

### PCA 

#### How does the PCA work and what is it doing to the data? 

<p>
PCA stands for Principal Component Analysis. Principal Component Analysis is designed to reduce time and the use or operation of computers resources when dealing with massive datasets. PCA reduces the number of information dimensions. The PCA algorithm transforms a dense set of variables into a smaller one without counteracting the loss of information as much as possible. Keep in mind that the PCA is mainly used for dimensionality reduction, not for visualization.
</p>

<p>
 PCA libaries can be different from each other due to parameters and attributes. For sklearn, it has has many parameters and attributes and one of the attributes that were used in this example is n_components. N_components will determine how many columns will be splitted into. If n_components equals to three, three columns will be created and hold the inputs.  
 </p>

![PCA Code Example](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/pca_example_1.png)

<p>
 
 </p>

#### PCA Outcome

<p>
In this case, PCA n_components equals point nine and it ended up having seventy-four columns. If n_components were equal to seventy-four, it will have the same outcome of seventy-four columns. However, the values will be barely different from each other. When n_components equal to .9, the first value is 0.0279317, and n_compnents equal to 74 has a value of 0.02793169. The value difference between the two is 0.00000001 (10 to the power of 7 x point 1). So either way will have the same output because of the number of columns produced. 
</p>

![PCA Variance Code Example](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/pca_example_3.png)

### t-SNE

<p>

</p>


<p>

</p> 

![]() <p> The following data frame is named <i> preview_get_dummies </i> displays two non-numeric columns. </p>

<p>

</p>

![]()

<p>

</p>

<p>

</p>

![]()

![]()

<p>

</p>

![]()

### LogisticRegression Model() and RandomForestClassifier() without StandardScaler


<p>

</p>

![]() 

<p>
 
</p>

<p>

</p>

![]()

<p>

</p>

### LogisticRegression Model() and RandomForestClassifier() with StandardScaler

#### How does the Standard scaler work and what is it doing to the data? 

<p> 
</p>

#### Example of Standard Scaler
<p>

</p>

<p>

  </p>

![]()

<p>

</p>

![]()

<p>

</p>

![]() 

<p>

</p>

### LogisticRegression Model() and RandomForestClassifier() Comparison

<p>

</p>

<p>
</p>



## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/samuelroiz/1af49ec9eea365bc845ba04c5071a976) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Samuel Roiz** - *Data clean, Analyzed Data, Math Model* - [Profile](https://github.com/samuelroiz)

See also the list of [contributors](https://github.com/samuelroiz) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/samuelroiz/1af49ec9eea365bc845ba04c5071a976) file for details

## Acknowledgments

* CryptoCare
* Advisory Services Team 
* USC Data Visualization
* CSUN Mathematics

