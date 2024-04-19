---
# Advisory Services Team Analyze

Created a new cryptocurrency investment portfolio for the Advisory Services Team of a financial consultancy. Clients are interested in offering a new cryptocurrency investment portfolio for their customers. The company, however, is lost in the vast universe of cryptocurrencies. The company asked to create a report that includes what cryptocurrencies are on the trading market and determines whether they can be grouped to create a classification system for this new investment.

The organization gave raw data to process it to fit the machine learning models. Using unsupervised learning and several clustering algorithms to explore the cryptocurrencies can be grouped with other similar cryptocurrencies. Create data visualization to share the findings with the investment bank and communicate with the non-data science audience.

## Data Clean Steps

The Advisory Services Team of a financial consultancy gave and created an unfinished data frame. The data frame has seven columns with one thousand two hundred fifty-two rows. The seven columns are named Unnamed: 0, CoinName, Algorithm, IsTrading, ProofType, TotalCoinsMined, and TotalCoinSupply.

![Given Dataframe](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/given_data_frame_from_crypto_company.png)

Data frames should not contain any null values or it will create conflicts with machine learning and plotting. A clean data frame is a data frame that does not include any null values. The given data frame has 508 null values in the column TotalCoinsMined and is the only column to contain null values.

![Check null val. in data frame](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/check_null_val_and_duplicates.png)

There are two options to approach this route. You can drop the TotalCoinsMined column since it is the only column that contains null values. The other option is to drop **ALL** null values in the data frame. In some cases, it is not advisable to drop the null values since they can be valuable information. In this case, they are not valuable.

![Null val. dropped](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/dropped_null_values.png)

It is important to double-check if the code went through. In this case, the code checks if the data frame has any null values. The microcode shows the following data frame does not have nulls then the argument worked.

![Check null val. in data frame again](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/check_null_val_and_duplicates_again.png)

The data frame column isTrading has two boolean values. If a row is true then cryptocurrency is still trading. Otherwise, it is not actively trading. Due to the confusion of boolean values, the column will convert from boolean to integer values to avoid conflicts. One will be actively trading, zero will be no trading. After converting the column, the data frame is filtered to rows containing one. The company only wants cryptocurrencies that are currently trading.

![Filter data frame of Crypto Currencies still trading](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/filter_dataframe_to_isTrading_1_only_aka_true_only.png)

The data frame has a column named TotalCoinsMined and has some empty values or zero. The company only wants cryptocurrencies that have coins mined more than zero. Total coins mined are essential to avoid scams and reduce the risk of counterfeiting, duplicates, or repeating the same coin more than once. In this case, the data frame will filter out cryptocurrencies rows that have not been mined.

![Filter data frame of Crypto Mined greater than zero](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/data_cleaning_and_filter/filter_dataframe_to_crypto_mined_greater_than_0.png)

The old data frame had seven columns with one thousand two hundred fifty-two rows. The clean data frame has four columns with five hundred thirty-two rows and active trading currencies. Columns Unnamed: 0, IsTrading, and CoinName are dropped for machine learning.

![Clean Data frame outcome](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/final_clean_data_outcome_from_crypto_company.png)

## Machine Learning Steps

### Get_dummies() Section

Here is an example of how `pd.get_dummies()` works.

![Get_Dummies() Part 1](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/example_get_dummies_part_1.png) 

The following data frame is named *preview_get_dummies* and displays two non-numeric columns.

Then apply the `get_dummies()` code:

```python


pd.get_dummies(preview_get_dummies)
```

![Get_Dummies() Part 2](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/example_get_dummies_part_2.png)

Now the *preview_get_dummies* data frame has numeric columns which means it meets the standards of machine learning models.

### Get Dummies Outcome

Apply the `get_dummies()` code to the data frame, and the data frame ends up having five hundred thirty-two rows and ninety-eight columns.

![Get Dummies Outcome](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/get_dummies_outcome.png)

### Standard Scaler Section

Standard Scaler uses the formula:

```plaintext
z = (x - u) / s
```

where `z` is the scaled data, `x` is the data to be scaled, `u` is the average of the training samples, and `s` is the standard deviation of the training samples. The average or mean is the sum of all data points divided by the number of data points. In this case, `u` will sum all the training samples and divide by the number of these samples. The standard deviation formula starts by taking the given values and placing them in one column. Square each value and place them in a second column. Find the sum of all values in the first column and square it. The value is divided by the number of data points in the first column and called this number `i`. Find the sum of all values in the created second column. Once you find the value, subtract it by `i`. The outcome will be divided by the number of data points minus one. This value leads to the `variance` of the sample and data. Finally, the variance will be square-rooted leading to the standard deviation of the data.

#### Example of Standard Scaler

To demonstrate the algorithm and how it functions, consider the dataset `{1,2,3,4,5}`. The dataset consists of 5 one-dimensional data points, and each data point has one feature. Now apply the standard scaler to the data. The dataset becomes `{-1.41, -0.71, 0, 0.71, 1.41}`.

The following example takes all of the data points and converts them to a closer range of 0 to 1. The standard scaler helps prevent outliers and keeps the data closer to each other rather than gaps.

### Standard Scaler Outcome

Apply the standard scaler code to the dummies' data frame to demonstrate the algorithm and drive the data closer to each other.

![Standard Scaler X Values](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/standard_scaler_x_val.png)

### PCA

#### How does the PCA work and what is it doing to the data?

PCA stands for Principal Component Analysis. Principal Component Analysis is designed to reduce time and the use or operation of computer resources when dealing with massive datasets. PCA reduces the number of information dimensions. The PCA algorithm transforms a dense set of variables into a smaller one, minimizing the loss of information as much as possible. Keep in mind that the PCA is mainly used for dimensionality reduction, not for visualization.

PCA libraries can be different from each other due to parameters and attributes. For sklearn, it has many parameters and attributes, and one of the attributes that were used in this example is `n_components`. `n_components` will determine how many columns will be split into. If `n_components` equals three, three columns will be created and hold the inputs.

![PCA Code Example](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/pca_example_1.png)

#### PCA Outcome

In this case, `PCA n_components` equals 0.9, and it ended up having seventy-four columns. If `n_components` were equal to seventy-four, it would have the same outcome of seventy-four columns. However, the values will be slightly different from each other. When `n_components` equals 0.9, the first value is 0.0279317, and `n_components` equal to 74 has a value of 0.02793169. The value difference between the two is 0.00000001 (10 to the power of 7 x 0.1). So either way will have the same output because of the number of columns produced.

![PCA Variance Code Example](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/pca_example_3.png)

### t-SNE

#### How does the t-SNE work and what is it doing to the data?

t-SNE stands for t-distributed stochastic neighbor embedding. The t-distributed is where

 a probability curve is generated, the stochastic is where results will be different each time, and neighbor embedding is where similar data points become neighbors. t-SNE functions similarly to PCA where t-SNE reduces a dataset’s dimensions to 2 or 3. The difference is t-SNE sorts unlabeled data into clusters and is mainly used to visualize data. The model parameters can produce different results drastically each time when run. This also applies to the visualizations that will not be the same. Reminder, the 's' in t-SNE means stochastic, which indicates randomness each time it runs. The visualization of the cluster sizes and distance between clusters do not necessarily represent actual sizes and distances.

t-SNE libraries can be different from each other due to parameters and attributes. For sklearn, it has many parameters and attributes. One of the parameters used in this example is perplexity. Perplexity will determine the number of neighbor inputs in the 2 or 3 columns.

![t-SNE Code Example](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/tsne_example_1.png)

#### t-SNE Outcome

For this case, t-SNE has perplexity equal to thirty and merged into two columns. After fitting data, t-SNE fitted along PCA values allowing its values to be in their shape. Then plotted into a scatter plot. The plot shows multiple clusters and an outlier. The clusters are numerous plots that are clumped and a single point. The outlier is in the corner likely caused by the outliers in the data. The data most likely made an error with an integer. However, maybe it is not because one of the most mined and popular currencies is Bitcoin. According to research, bitcoin makes up at least sixty percent of the crypto market. From the following graph, there are four to five clusters.

![t-SNE plot prex. 30](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/tsne_example_3.png)

### KMeans

#### How does the KMeans work and what is it doing to the data?

K-means is an unsupervised learning algorithm used to identify clusters and group the data into groups based on the distance measured to a centroid. A centroid is a data point that is the arithmetic mean position of all the values in a cluster. When the code is running, it will take steps. The first step is to randomly initialize the k starting centroids. The second step is to assign each data point to the nearest centroid. The third step is to recompute the centroids again as the average of the data plots assigned to the individual cluster. The final step is to repeat the first, second, and third steps again until the algorithm is finished. After applying the algorithm, the objective is to plot and find a slope in the graph. The slope should look like the second dip converging to a value. This is known as inertia.

#### KMeans Outcome

![KMeans Code Example](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/k_means_example_1.png)

![Elbow Plot](https://github.com/samuelroiz/Advisory_Services_Team_Analyze/blob/main/images/machine_learning/k_means_example_2.png)

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/samuelroiz/1af49ec9eea365bc845ba04c5071a976) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

- **Samuel Roiz** - *Data clean, Analyzed Data, Math Model* - [Profile](https://github.com/samuelroiz)

See also the list of [contributors](https://github.com/samuelroiz) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/samuelroiz/1af49ec9eea365bc845ba04c5071a976) file for details.

## Acknowledgments

- CryptoCare
- Advisory Services Team
- USC Data Visualization
- CSUN Mathematics
