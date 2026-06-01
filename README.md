# A More Accurate Algorithm Comparison through A/B Testing using Offline Evaluation Methods

This repository contains the code for the experiments in "A More Accurate Algorithm Comparison through A/B Testing using Offline Evaluation Methods".

## Requirements

uv (https://docs.astral.sh/uv/) are required. Run the following to set up the environment:

```bash
uv sync
```

## Dataset Setup

This project uses the [KuaiRec](https://kuairec.com/) dataset, a real-world video viewing dataset collected from the recommender system of the video-sharing mobile app Kuaishou.

### Download

1. Visit [https://kuairec.com/](https://kuairec.com/) and download the dataset.
2. Place the `small_matrix.csv` under the `data/` directory:

```
kdd2026-mid/
└──data/
    └── small_matrix.csv        # Fully-observed user-item matrix
```

# Experiments

Run the notebooks/*.ipynb

## Abstract

A/B testing is the gold standard for selecting better algorithms in online services. While offline evaluation has attracted attention as a safer alternative due to the high experimental costs and the potential risk of degrading user experience and revenue in A/B testing, it is widely recognized that the estimation accuracy of offline evaluation is substantially lower than that of A/B testing. As a result, final decisions on algorithm selection are typically made through A/B testing.
Contrary to this conventional view, we reveal a counterintuitive phenomenon in which A/B testing can produce a higher algorithm selection error rate than offline evaluation. This occurs because the sample mean estimator used in A/B testing does not induce positive correlation, which plays a crucial role in reducing critical selection errors, namely underestimating the truly superior algorithm and overestimating the truly inferior one. In contrast, offline evaluation methods unintentionally generate this beneficial correlation by relying on shared offline data when estimating and comparing the performance of multiple algorithms.
Building on this insight, we propose a novel estimator that intentionally induces positive correlation to improve algorithm selection in A/B testing. The key idea is to introduce a hypothetical middle algorithm and to estimate the performance difference between algorithms A and B in a stepwise manner, first between A and the middle algorithm and then between the middle algorithm and B, using shared data at each step. This approach enables the application of offline evaluation techniques in each step, thereby inducing positive correlation and reducing critical selection errors. Furthermore, we derive the optimal middle algorithm regarding the resulting variance and analyze its advantages over existing methods through bias-variance analysis. Experiments on real-world data demonstrate that the proposed estimator achieves the same selection error rate as existing approaches while using only one half of the A/B testing data, indicating a twofold improvement in sample efficiency.

## Citation
```
hoge
```