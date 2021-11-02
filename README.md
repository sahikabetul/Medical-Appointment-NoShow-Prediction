# Medical-Appointment-NoShow-Prediction | Which Patient will not Come to the Appointment?
## Solution to a daily life problem guided by 'Medical Appointment No Shows' data from 2016

<img src="https://img.freepik.com/free-vector/set-doctor-patient-cartoon-characters_36082-522.jpg?size=626&ext=jpg" width="800px" height="auto">

Not coming to examination appointments is a negative situation for the health system as well as the patient.

Patients who do not come to their appointment on time cause an interruption in the scheduling process. If they do not come at all, they waste the doctor's time, and if they come late, they cause the doctor to do extra work. In both cases, the efficiency of the health system reduces.

There are monetary costs associated with no-shows. A study highlighted in Health Management Technology found that missed appointments in the U.S. costs the industry an astounding $150 billion dollars. On average, each physician loses $200 per unused time slot.

## Motivation for the Project
Investigate the reason why some patients do not show up to their scheduled appointments and find the probability of missing the appointment.

## Summary of the Results of the Analysis
Best model: Random Forest Classifier , f1-score: 0.88, Accuracy: 0.79

According to trained models feature importance, we could see that Gender, Scholarship, Hypertension and Diabetes are some of the top features that would help us determine if the patient who has taken an appointment will show/no-how.

## Libraries Used

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report,confusion_matrix,f1_score, make_scorer
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

## Files in the Repository
**KaggleV2-May-2016.csv:** the used dataset.

**Medical Appointment Show-Up Prediction.ipynb:** the notebook of the project.

## Medium Blog Post
Lets go to my Medium blog post: 
[Which Patient will not Come to the Appointment? | End-to-End Machine Learning Code Example](https://medium.com/@sahika.betul/which-patient-will-not-come-to-the-appointment-end-to-end-machine-learning-code-example-e952f65888ac)
