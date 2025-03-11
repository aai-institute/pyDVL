---
title: Applications of data valuation
alias:
  name: data-valuation-applications
---

# Applications of data valuation

Data valuation methods can improve various aspects of data engineering and
machine learning workflows. When applied judiciously, these methods can enhance
data quality, model performance, and cost-effectiveness.

However, the results can be inconsistent. Values have a strong dependency
on the training procedure and the performance metric used. For instance,
accuracy is a poor metric for imbalanced sets and this has a stark effect
on data values. Some models exhibit great variance in some regimes
and this again has a detrimental effect on values. See
[Problems of data values][problems-of-data-values] for more on this.

Here we quickly enumerate the most common uses of data valuation. For a
comprehensive overview, along with concrete examples, please refer to the
[Transferlab blog post]({{ transferlab.website }}blog/data-valuation-applications/)
on this topic.

## Data engineering

Some of the promising applications in data engineering include:

- Removing low-value data points to increase model performance.
- Pruning redundant samples enables more efficient training of large models.
- Active learning. Points predicted to have high-value points can be prioritized
  for labeling, reducing the cost of data collection.
- Analyzing high- and low-value data to guide data collection and improve
  upstream data processes. Low-value points may reveal data issues to address.
- Identify irrelevant or duplicated data when evaluating offerings from data
  providers.

## Model development

Some of the useful applications include:

- Data attribution for interpretation and debugging: Analyzing the most or least
  valuable samples for a class can reveal cases where the model relies on
  confounding features instead of true signal. Investigating influential points
  for misclassified examples highlights limitations to address.
- Sensitivity / robustness analysis: [@broderick_automatic_2021] shows that
  removing a small fraction of highly influential data can completely flip model
  conclusions. This can reveal potential issues with the modeling approach, data
  collection process, or intrinsic difficulties of the problem that require
  further inspection.
- Continual learning: in order to avoid forgetting when training on new data,
  a subset of previously seen data is presented again. Data valuation can help
  in the selection of the most valuable samples to retain.

## Attacks

Data valuation techniques have applications in detecting data manipulation and
contamination, although the feasibility of such attacks is limited.

- Watermark removal: Points with low value on a correct validation set may be
  part of a watermarking mechanism.
- Poisoning attacks: Influential points can be shifted to induce large changes
  in model estimators.


## Data markets

Additionally, one of the motivating applications for the whole field is that of
data markets, where data valuation can be the key component to determine the
price of data.

Game-theoretic valuation methods like Shapley values can help assign fair prices,
but have limitations around handling duplicates or adversarial data.
Model-free methods like LAVA [@just_lava_2023] and CRAIG are
particularly well suited for this, as they use the Wasserstein distance between
a vendor's data and the buyer's to determine the value of the former. 

However, this is a complex problem which can face simple practical problems like
data owners not willing to disclose their data for valuation, even to a broker.
