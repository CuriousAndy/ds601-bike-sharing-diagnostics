# Diagnosing and Remedying Linear Model Failures in Daily Bike-Sharing Demand

DS601 Statistical Learning, Group 2: Zhang Enhao, Jiang Liuru, Su Liangyu, Ma Tianjiao

Working draft. Each section names who writes it. The leader merges everything into one voice at the end.

## 1. Introduction and research question

A very practical problem in running a bike-sharing system is how many bikes are needed each day, and how to move bikes between stations in advance. If demand is predicted too low, some stations can run out of bikes quickly; if it is predicted too high, there are extra rebalancing costs. Linear regression is a common starting point for this kind of prediction, because the model is simple and makes it fairly easy to explain how different weather and calendar factors relate to demand. We use the daily data from Capital Bikeshare in Washington, D.C. for 2011-2012, 731 days in total.

We mainly want to answer three questions. First, how much of the day-to-day variation in rentals can weather and calendar variables explain? Second, which assumptions of ordinary OLS do not hold in this data? Third, and this is what the report cares about most: after we apply remedies for these problems, does the model actually become more reliable, or does only its R² on the training data look higher? To answer this, we first run model diagnostics, then try the matching remedies, then check out-of-sample prediction with time-ordered cross-validation, and finally audit the model once on a fully held-out test set. We also check whether the prediction errors differ clearly across day types and weather conditions.

## 2. Data

[TODO-LATER · Jiang Liuru, from A3] Source, licence, 731 days, train/test split, temperature conversion, external event table.

## 3. Methods

### 3.1 Base model

[TODO-LATER · Su Liangyu, from B3]

### 3.2 Exclusion rules and influence adjudication

[TODO-LATER · Ma Tianjiao, from C3]

### 3.3 Remedies

[TODO-LATER · Ma Tianjiao, from C3]

### 3.4 Cross-validation

We use RMSE as the main metric for comparing models, measured in rentals per day. One advantage of RMSE is that its unit is easy to understand, and it gives more weight to the days when the model makes a very large error. We did not shuffle the data and run ordinary cross-validation, because this data has a clear time structure and neighbouring dates tend to be similar. The Durbin-Watson statistic is only 1.24, which also shows positive autocorrelation in the residuals. If neighbouring dates are mixed randomly into the training and validation sets, the results can easily be too optimistic.

We therefore validate by moving forward in time, always using only past data to predict the next period. The three validation windows cover February-April, April-June and June-August 2012. In our first design, the training data for the first fold stopped at the end of 2011, and the first fold's RMSE came out at 1,871. When we checked, `yr` was 0 for every row in that training data, so the model had never seen 2012 and could not learn that overall demand rose in 2012. We changed the first fold's training data to end on 31 January 2012, which keeps a full 12-month seasonal cycle and also lets the model see some 2012 data.

Exactly the same exclusion rules are used in every fold. More importantly, every later model comparison, including month vs season and the comparisons between remedies, uses this same time-ordered cross-validation, so the results of different models can be compared directly. Shuffled 5-fold CV is only a reference point, to show how much shuffling time-ordered data understates the error; it is never used to pick the final model. The design has one known limit: the validation windows are mainly in February-August, while the final test set is in August-December, so the seasons do not fully overlap. We check this again in the test-set analysis with RMSE by month.

### 3.5 Fairness analysis

Besides overall RMSE, we also want to know whether the model keeps doing worse on certain kinds of days. This matters in practice: if the model systematically under-predicts demand on one kind of day, the operator is more likely to be short of bikes on those days. We group mainly by day type (working day, weekend and holiday) and by the day's weather. We chose these variables because they are basically known before the day starts, so an operator can really use them to decide in advance.

We did not group by casual share. Casual share is calculated from the rentals that actually happened that day, so it is only known after the outcome, and using it to define groups before prediction brings in information from after the fact. Every error used here comes from a true out-of-sample prediction: the day's actual data was not used to train the model that predicts it. The analysis covers February to August 2012, 188 prediction days in total. Residuals are defined as actual minus predicted, so a positive residual means the model under-predicts demand; if a group keeps showing positive residuals, the model could leave the operator short of bikes on that kind of day.

## 4. Results

### 4.1 Base model diagnostics

[TODO-LATER · Su Liangyu]

### 4.2 Remedies: before and after

[TODO-LATER · Ma Tianjiao]

### 4.3 Forward-chaining vs shuffled CV

[TODO-LATER · Zhang Enhao, after 26 Oct]

### 4.4 Fairness

[TODO-LATER · Zhang Enhao, after 26 Oct]

### 4.5 Test-set audit and final model

[TODO-LATER · Zhang Enhao, after the audit]

## 5. Discussion and limitations

[TODO-LATER · Zhang Enhao]

## 6. Conclusion

[TODO-LATER · Zhang Enhao]

## References

Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors and background knowledge. Progress in Artificial Intelligence, 2(2-3), 113-127. https://doi.org/10.1007/s13748-013-0040-3

Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894

James, G., Witten, D., Hastie, T., Tibshirani, R., & Taylor, J. (2023). An Introduction to Statistical Learning with Applications in Python. Springer.

## Declaration of generative AI and AI-assisted technologies in the project

[TODO-LATER · Zhang Enhao, from AI_USAGE.md]
