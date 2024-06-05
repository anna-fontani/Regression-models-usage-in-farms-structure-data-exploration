# Regression Models Usage in Farms Structure Data Exploration

**Task: Explore and analyze factors influencing farms performance in European Union using ML algorithms.**

Previous part of this project: Data cleaning and preprocessing: https://github.com/anna-fontani/Data-cleaning-and-preprocessing
Data was explored and preprocessed, including missing values imputation (IterativeImputer from sklearn library).

Dataset comprising of EU countries and their agriculture-related features was used to perform regression modelling of dependent variable – financial output of agricultural sector. 

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/08773897-cd54-4185-af1b-b6b66e90073d)

Figure 1. Density plots of main features

## Scaling 

Difference in magnitude in features can create problems for ML algorithms. Algorithms that use gradient descent require scaling due to possible step size differences. Our dataframe contained data with different magnitude and required it to be addressed. When comparing normalization and standardization, standardization is less sensitive to outliers, preserves relationships between data, and can be used for data with unknown distribution (Bhandari, 2024). For these reasons, StandardScaler was applied to the dataframe.

![image](https://github.com/anna-fontani/Data-cleaning-and-preprocessing/assets/149007143/a3cd53a3-4655-4f55-9cf3-1c2e6afc881f)

Table 1. Scaling results.

## Regression models

Following models were applied to address research question: Linear regression, Ridge regression, Random forest regression.

Experiments were run with 3 features (farms number, country area, used agricultural area) and 8 features (farms number, country area, used agricultural area, total labour, nonfamily labour, managers with basic training, managers with only practical training, managers with full training). 

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/162a0d54-6a73-4c63-b285-ed4bb0066dd8)

Table 2. Results of Linear regression experiments with 3 and 8 features.

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/f0cafa32-1bca-48fd-87ad-4b27f65c756d)

Table 3. Results of Ridge regression experiments with 3 features.

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/5b564cb7-0a78-47f8-9258-c073de0a9c18)

Table 4. Results of Ridge regression experiments with 8 features.

Ridge regression experiments were performed for parameter alpha = 0.1, alpha = 10, and without setting alpha. The best result was recorded for 80% / 20% split, no alpha for 3 features - training set score 0.77 and test set score 0.82. With 8 features, best result recorded with 80% / 20% split, alpha=0.1, training set score 0.91 and test set score 0.74.

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/f46ca2f8-cbd6-459e-aca0-14285e918a99)

Table 5. Results of Random forest experiments with 3 and 8 features.
 
Best results of Random forest algorithm were recorded for 80% / 20% split. Experiments with 8 features showed better result, R2 train: 0.96, test: 0.86. 

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/6d3fbf9d-2899-4c5f-a3e3-031d6ad270bd)

Figure 2. Random forest results with 8 features and 80% / 20% split.

Summing up, among three models and various parameters reviewed, best results of regression task for financial output variable were demonstrated by Random Forest method for 8 independent features with resulting R2 train: 0.96, test: 0.86.

## Parameters Tuning 

GridSearchCV was applied to tune parameters of Ridge model with following parameters. Cross validation parameter folds = 5.

![image](https://github.com/anna-fontani/Regression-models-usage-in-farms-structure-data-exploration/assets/149007143/767537fb-3a5c-4571-8642-ad931201c2c5)

Figure 3. GridSearchCV parameters for Ridge regression model tuning.  

Complete results of 5 folded method application are preview demonstrated below.

Table 10. GridSearchCV results preview.




## References

- Bhandari, A. (2024). Feature Scaling: Engineering, Normalization, and Standardization. [online] Analytics Vidhya. Available at: https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization.
