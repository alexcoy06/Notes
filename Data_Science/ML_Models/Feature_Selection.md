# Feature Selection Methods for Machine Learning with Python Examples

These examples demonstrate nine common feature selection techniques to help reduce model complexity and improve predictive performance. Each snippet shows how to implement a method using Python libraries such as scikit-learn (and mlxtend for sequential selection).

---

## Filter Methods

**1. Variance Threshold**  
Removes features with little variability that may not be informative. This simple method is effective for eliminating constant or near-constant features.

```python
from sklearn.feature_selection import VarianceThreshold

# Assuming X is your feature matrix
selector = VarianceThreshold(threshold=0.1)
X_reduced = selector.fit_transform(X)
```

**2. Correlation Analysis**  
Identifies highly correlated features that can cause redundancy in the model. Drop one feature from each pair of features exceeding a correlation threshold.

```python
import pandas as pd
import numpy as np

# Assume df is a DataFrame with numeric features
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
df_reduced = df.drop(columns=to_drop)
```

**3. Statistical Tests (Chi-Square)**  
Evaluates the relationship between categorical features and the target variable. Chi-square tests can help select features that are statistically significant.

```python
from sklearn.feature_selection import SelectKBest, chi2

# X should contain non-negative features (e.g., counts) and y the target variable
selector = SelectKBest(score_func=chi2, k=5)
X_new = selector.fit_transform(X, y)
```

---

## Wrapper Methods

**4. Recursive Feature Elimination (RFE)**  
Iteratively removes the least important features based on a model's performance. RFE helps in selecting features that contribute most to predicting the target variable.

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)
```

**5. Forward Selection**  
Starts with no features and adds one feature at a time that improves the model performance. This method is useful when you want to build a model from a minimal set of features.

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
sfs_forward = SFS(lr,
                  k_features=5,
                  forward=True,
                  floating=False,
                  scoring='accuracy',
                  cv=5)
sfs_forward = sfs_forward.fit(X, y)
X_forward = sfs_forward.transform(X)
```

**6. Backward Elimination**  
Begins with all features and removes the least important one at each step. This approach is ideal when starting with a comprehensive set and narrowing it down based on performance.

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
sfs_backward = SFS(lr,
                   k_features=5,
                   forward=False,
                   floating=False,
                   scoring='accuracy',
                   cv=5)
sfs_backward = sfs_backward.fit(X, y)
X_backward = sfs_backward.transform(X)
```

---

## Embedded Methods

**7. Regularization Techniques (LASSO)**  
Uses L1 regularization to shrink less important feature coefficients to zero, effectively performing feature selection. LASSO is particularly useful when you expect only a subset of features to be useful.

```python
from sklearn.linear_model import Lasso
import numpy as np

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
important_features = np.where(lasso.coef_ != 0)[0]
```

**8. Tree-Based Feature Importance**  
Leverages decision trees or ensemble methods like Random Forests to rank features by importance. This method is useful for capturing non-linear relationships in the data.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_
```

**9. Regularized Models (Logistic Regression with L1 Penalty)**  
Incorporates feature selection into model training by applying L1 regularization within logistic regression. This approach not only builds the model but also helps in identifying the most significant features.

```python
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression(penalty='l1', solver='liblinear')
model.fit(X, y)
important_features_lr = np.where(model.coef_[0] != 0)[0]
```

These examples provide a practical starting point for feature selection in your machine learning projects, helping you build more efficient and interpretable models.
