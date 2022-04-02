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

</p>

![]()

## Machine Learning Steps

<p>

</p>

![]()

![]()

<p>
 
</p>

<p>

</p>

#### <u> pd.get_dummies() </u>

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

