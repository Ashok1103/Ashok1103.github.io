---
title: "What a 0.59 AUC Score taught me more than 0.95 ever could"
date: 2026-05-12
categories: [Machine Learning, Data Science]
tags: [random-forest, scikit-learn, feature-engineering,
       class-imbalance, k-means, pandas, business-ml]
---

Most ML tutorials end with a satisfying accuracy number. The model works,
the curve looks good, and everyone moves on. Real business data does not
always cooperate. This post is about a project where my best model scored
0.59 AUC, and why that result was more useful to the organization than a
clean 0.90 would have been.

---

## The Problem

During my co-op at a professional education organization, I was asked to
answer one question: **can we predict, at the moment of enrollment, which
learners are likely to drop out of a program?**

If yes, the team could intervene early with targeted support. If no, that
answer is equally important because it points to exactly what data needs
to be collected that currently is not. Either outcome has direct business
value.

---

## The Dataset

The dataset was built by merging several internal enrollment files spanning
multiple programs and years into a single analytical table of several
thousand records. Each record represented one learner enrollment and
included fields like program name, geographic location, job title,
employer, and enrollment status.

The target variable was binary: **Enrolled (1)** versus
**Dropped or Transferred (0)**. The class distribution was heavily
skewed, with enrolled learners significantly outnumbering dropouts. This
kind of imbalance is common in real retention datasets and needs to be
handled explicitly before modelling, otherwise the model learns to just
predict the majority class every time.

Three columns were dropped from the feature set before touching a model:

- One field had over 60% missing values, too sparse to be reliable
- A second had both high missingness and extremely high cardinality,
  making it noise after encoding
- A third was superseded by an engineered column described below

---

## Feature Engineering: Taming High Cardinality

The raw job title column had hundreds of unique values. Direct one-hot
encoding would have created hundreds of sparse binary columns, most
appearing only once or twice in the dataset, a fast path to overfitting.

Instead, I engineered a **Job Family** column by stripping seniority
prefixes (Senior, Junior, Lead, Principal) and normalizing the remaining
role name. "Senior Data Analyst", "Sr. Data Analyst", and "Lead Data
Analyst" all became "Data Analyst". This reduced cardinality from
hundreds of unique titles to a manageable set of professional role
categories, then capped at the top 30 most frequent values with
everything else collapsed into "Other".

The high null rate in job title was itself analytically meaningful.
Learners who did not list a professional title behaved differently from
those who did, so nulls were filled with "Unknown" and treated as a
distinct group rather than dropped.

![K-Means elbow plot](/assets/img/posts/kmeans_elbow.png)
*The elbow method across K=2 to K=10. The near-linear decline with 
no sharp bend suggests the data has no strongly separated natural 
clusters, consistent with the weak predictive signal found in 
classification.*
---

## The Models

I used a stratified 80/20 train/test split to preserve the class
imbalance ratio in both subsets, then trained two models:

**Logistic Regression (baseline):** configured with
`class_weight='balanced'` to compensate for the skewed split, and
`max_iter=1000` to ensure convergence across the 64 encoded feature
columns produced by one-hot encoding.

**Random Forest (main model):** same class weighting, with the added
benefit of feature importance scores showing which variables the model
actually relied on.

| Model | AUC-ROC | Accuracy | Dropout Recall |
|---|---|---|---|
| Logistic Regression | 0.59 | 61% | 49% (28/57) |
| Random Forest | 0.58 | 65% | 44% (25/57) |

Neither model crossed 0.60 AUC. On a balanced random baseline you would
expect 0.50. These models are barely better than guessing on the dropout
class.

![Logistic Regression confusion matrix](/assets/img/posts/lr_confusion_matrix.png)

![Random Forest confusion matrix](/assets/img/posts/rf_confusion_matrix.png)
*Logistic Regression catches 28 of 57 dropouts (49% recall). Random 
Forest catches 25 of 57 (44% recall) despite higher overall accuracy, 
because it leans harder toward the majority enrolled class. Neither 
result is a tuning problem. Both are hitting the same data ceiling.*

---

## The Real Finding

A sub-0.60 AUC is not a modelling failure. It is a **data ceiling**.

The features available at enrollment time simply do not contain enough
signal to reliably separate future dropouts from completers. The model
extracted everything it could from the available information and found
almost nothing predictive.

The Random Forest feature importance scores confirmed this. The most
influential features were broad structural variables like program type
and geographic region. No demographic or occupational feature at
enrollment consistently distinguished someone who would drop out from
someone who would complete.

This is actually a concrete, actionable result. It tells the organization
exactly what data to start collecting: **engagement signals during the
program** rather than demographic signals before it starts. A learner's
job title at enrollment tells you almost nothing about whether they will
finish. How they engage in week two probably tells you a great deal.

![Random Forest feature importance](/assets/img/posts/rf_feature_importance.png)
*Course type and province are the top predictors, both broad structural 
variables. The Job Family Unknown group (learners with no listed job 
title) ranks fourth, higher than any specific professional role. 
No individual occupational feature carries meaningful weight.*

---

## Bonus: Learner Segmentation via K-Means

While the classification task hit a ceiling, clustering revealed something
useful. Applying K-Means to the enrollment data identified several
distinct learner profiles with meaningfully different characteristics
across geography, seniority level, and professional background.

These segments are directly actionable for marketing and program design,
even when dropout prediction is not reliable. Knowing *who* enrolls,
in structured clusters rather than raw counts, helps a team make better
decisions about where to focus acquisition efforts and how to tailor
program communication.

---

## What This Project Taught Me About Real ML

Clean benchmark datasets are designed to be solvable. A ResNet on
ImageNet will hit 90%+ if you do the engineering correctly. Real
business data is not designed for anything. It reflects what an
organization happened to collect, which is rarely the data you would
have chosen if you knew what question you were going to ask.

The most important skill in applied ML is not picking the right model.
It is knowing when a poor result is telling you something true about
the data rather than something fixable about the model. In this case,
the 0.59 AUC is not a number to hide. It is the finding.

The methodology and code are on
[GitHub](https://github.com/Ashok1103/Machine-Learning-Portfolio/blob/main/WatSPEED%20Enrolment%20Prediction).
Note that the dataset itself is not included in the repository as it
contains proprietary organizational data.
