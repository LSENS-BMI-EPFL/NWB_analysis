# SWR Decoding Project – Analysis Log

Author:
Start date: 2026-02-16

---

# Project Overview

## Motivation

The goal of this project is to investigate whether **hippocampal sharp-wave ripples (SWR) recorded in CA1 contain information about sensory stimuli presented during behavioral trials**.

Based on insights from the literature and discussions within the lab, we hypothesize that **population activity during ripple events may reflect task-related neural representations**, potentially allowing discrimination between different stimulus types.

Specifically, we aim to distinguish between:

* **auditory trials**
* **whisker trials**
* **no stimulus trials**

using population activity extracted from spike recordings.

---

# Dataset Exploration

## Data source

The dataset was provided by **Robin**.

For each mouse, a table contains **trial-level neural and behavioral data**.

Each row corresponds to a trial and includes:

* spike times around **sensory events**
* spike times around **ripple events**
* behavioral metadata:

  * trial type
  * reward condition (R+ / R−)
  * lick flag
  * context
  * ripple counts per trial

Spike activity is stored as **lists of spike times**, organized as **list-of-lists**, where each inner list corresponds to the spike times of a neuron.

---

# Feature Construction

## Population Vectors

Spike times were converted into **population vectors** by counting spikes within defined time windows.

For each trial:

```
trial_vector = [n1_spikes, n2_spikes, ..., nN_spikes]
```

Each vector represents the **ensemble neural activity across recorded neurons**.

---

## Time Windows Used

Two time windows were defined:

### Sensory window

0 – 50 ms after stimulus onset

Purpose:
capture **stimulus-driven neural activity**

### Ripple window

−50 ms to +50 ms around ripple onset

Purpose:
capture **neural activity surrounding SWR events**

---

# Initial Decoding Approach

## Method

We applied **Linear Discriminant Analysis (LDA)** to explore whether population activity separates trial types. The LDA try to minimize the ratio by minimzing the variance inside the classes and maximzing the one between the classes. 

Procedure:

1. Train LDA using **sensory population vectors**
2. Project **ripple population vectors** into the LDA space
3. Visualize projections using LD1 and LD2

Goal:
test whether **SWR activity reflects stimulus-related neural representations**.

---

# Early Observations

Initial LDA projections were **not strongly convincing**.

Observed patterns:

* clusters corresponding to trial types were **not clearly separated**
* ripple projections showed **substantial overlap between trial types**

Possible explanations:

* weak signal in ripple activity
* insufficient preprocessing
* differences between sensory and ripple representations.

---

# Alternative Analyses

To further investigate the structure of the data, additional decoding approaches were tested.

### Sensory-only decoding

LDA trained and projected using only **sensory population vectors**.

Result:

* moderate separation between trial types
* some visible clustering.

### Ripple-only decoding

LDA trained and projected using only **ripple population vectors**.

Result:

* clusters appeared in some mice
* ripple activity may still contain informative structure.

---

# Baseline Subtraction

To reduce background firing variability, a **baseline subtraction step** was introduced.

The objective was to remove baseline firing levels before computing population vectors.

After applying baseline subtraction:

* LDA separation improved in several mice
* projections became easier to interpret.

---

# Regional Comparison

To explore regional differences, analyses were performed for:

* **CA1**
* **SS1 (primary somatosensory cortex)**

---

# Visualization Improvements

To better interpret the results, plots were refined:

* consistent color coding for **trial types**
* additional colors to distinguish **reward conditions (R+ vs R−)**

This allowed clearer visualization of patterns in LDA space.

---

# Normalization Improvments 

As the measures were done on simultaneously several neurons, some of them could have a higher spike intensity naturaly, thus to avoid that these neurons dominate the other we apply a standardization to the data before fitting the LDA model

# Preliminary Observations

Across several mice:

* stronger separation appears in **R− trials**
* weaker or inconsistent separation in **R+ trials**

This effect is particularly visible in **ripple-related LDA projections**.

This may suggest that **reward condition influences the structure of ripple-related neural activity**.

---

# Current Status (after ~3 weeks)

Period covered:

2026-02-16 → 2026-03-08

Completed steps:

* dataset exploration
* population vector construction
* LDA decoding implementation
* sensory vs ripple comparisons
* baseline subtraction implementation
* visualization improvements
* preliminary regional comparison.

---

# Current Limitations

Current limitations include:

* reliance on **visual inspection of LDA projections**
* lack of quantitative evaluation metrics embedding all mice results
* possible **class imbalance**

---

# Next Steps

Planned analyses:

* compute **quantitative separation metrics**

  * centroid distances

---

# Week 4

## Quantitative evalutaion metrics embedding all mice results 

* In order to compare and make a summary for each mice, the trial type centroids their euclidian ditance were computed, in order to see quantitatively the degree of separtation between the trial classfications. 

### Method

The follwing distance was computed: No stim - whisker, No stim - auditory, whisker-auditory. Then, we plot the boxplot distribution of each distances combination type.

### Result

As observed with the 2D LDA plots, among all mice the  distibution of the distances between the centroids of auditoy and whisker trials computed from the LDA projection coordinates of the population vectors in the CA1 during the ripples are bigger for the non-rewarded mice than the rewarde mice. The population vectors are also quiet separable in the Ca1 buzt this time during the sensory input.  

### Interpretation

This clear separation between the type of trial obtained from the population vector of hippocampal neurons during ripples is quite impressive. Indeed, it would mean that activity during ripples points to the current trial and therefore to the sensory input that has just taken place. Furthermore, the greater separation seen in unrewarded mice could imply that they interpret the characteristics between an auditory stimulus and a tactile stimulus more clearly. This interpretation could then be reinforced by the fact that the tactile sensory input, compared to the auditory input received during their training, is not rewarded. 

### Next Step

In order to confirm that the 
