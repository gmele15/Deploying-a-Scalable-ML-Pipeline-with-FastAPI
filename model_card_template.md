# Model Card

## Model Details

### Overview

This is a scikit-learn Random Forest Classifier used to determine whether an individual has an inome over $50,000 based on demographics found in the UCI Census Income Dataset.  This is not recommended to be used as a production model, but was produced with the intention of exploration.  Default parameters for the RFC were used to produce this model.  This model was developed in a colaboration between Udacity and Gumaro Melendez.

## Intended Use

This model is not recommended or even suitable for real world applications.  This model was inted to be used as a tool for exploration and learning.

## Training Data

The model was trained using 66% of the total data set.  The data set includes categorical variables `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`.
With contiuous variables `age`, `fnlgt`, `education-num`, `capital-gain,` `capital-loss`, and `hours-per-week`.

## Evaluation Data

The remaining 33% of the data was used for testing.

## Metrics
The following metrics and measurements were observed from model evaluation:
Precision: 0.7226 | Recall: 0.6161 | F1: 0.6651

The model performed relatively similarly across all slices of categories producing a variance of +-0.01.

## Ethical Considerations

There were some under represented categories within the data.  Namely within Native Country and Race.  Although the model appears to be conservative in it's predictions, the imbalance in data required that more data and careful attention to the under represented groups be exercised.

## Caveats and Recommendations

The data used to train did not provide any information about location.  It is recommended that extreme caution is used and noted when exploring this model in different areas that may have a higher or lower than average PCI.
