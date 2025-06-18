**Objectives:** 
**Hyperparameter Optimization:** Using Monte Carlo Method to randomly sample hyperparametres instead of complicated grid researech methods. 
**Uncertainity Quantification:** Assessing variability of performance of the model across different test splits and parameter combinations. 
**Risk Evaluation:** Providing Confidence intervals and probability distributions for model accuracy. 

**Import Libraries:** 
```import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt```
**Define hyperparameter space:** 
'''n_simulations = 100
param_dist = {
    'n_estimators': np.random.randint(50, 300, n_simulations),
    'max_depth': np.random.randint(3, 15, n_simulations),
    'min_samples_split': np.random.uniform(0.01, 0.2, n_simulations)
}''' 
**Simulation:**  
'''results = []
for i in range(n_simulations):
    # Random train-test split (different each iteration)[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = RandomForestClassifier(
        n_estimators=param_dist['n_estimators'][i],
        max_depth=param_dist['max_depth'][i],
        min_samples_split=param_dist['min_samples_split'][i],
        random_state=42
    )
    model.fit(X_train, y_train)''' 
**Analyzing Results:** 
'''mean_acc = np.mean(results)
std_acc = np.std(results)
conf_95 = np.percentile(results, [2.5, 97.5])
print(f"Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
print(f"95% Confidence Interval: {conf_95}")''' 
**Plotting Distribution:** 
'''plt.figure(figsize=(10, 6))
plt.hist(results, bins=20, edgecolor='k', alpha=0.7, color='green')
plt.axvline(mean_acc, color='red', linestyle='--', label='Mean Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Range')
plt.title('Monte Carlo Simulation')
plt.legend()
plt.show()''' 
**Mean Accuracy:** 0.821 ± 0.018
**95% Confidence Interval:** [0.79703821 0.85201104]
![Alt text](https://drive.google.com/file/d/1mRAQHVJy0UiyAvuTjz_R8cWCeTQ8Kclh/view?usp=drive_link) 












