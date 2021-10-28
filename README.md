# Medical-Appointment-NoShow-Prediction | Which Patient will not Come to the Appointment?
## Solution to a daily life problem guided by 'Medical Appointment No Shows' data from 2016
Not coming to examination appointments is a negative situation for the health system as well as the patient.

Patients who do not come to their appointment on time cause an interruption in the scheduling process. If they do not come at all, they waste the doctor's time, and if they come late, they cause the doctor to do extra work. In both cases, the efficiency of the health system reduces.

There are monetary costs associated with no-shows. A study highlighted in Health Management Technology found that missed appointments in the U.S. costs the industry an astounding $150 billion dollars. On average, each physician loses $200 per unused time slot.

<img src="https://img.freepik.com/free-vector/set-doctor-patient-cartoon-characters_36082-522.jpg?size=626&ext=jpg" width="800px" height="auto">

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
import plotly.graph_objs as go
import datetime
from time import strftime
import statistics

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,f1_score
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.ensemble import GradientBoostingClassifier);
```

## Files in the Repository
**KaggleV2-May-2016.csv:** the used dataset.
**Medical Appointment Show-Up Prediction.ipynb:** the notebook of the project.

## Medium Blog Post
Lets go to my Medium blog post: 
[Which Patient will not Come to the Appointment? | End-to-End Machine Learning Code Example](https://medium.com/@sahika.betul/which-patient-will-not-come-to-the-appointment-end-to-end-machine-learning-code-example-e952f65888ac)
