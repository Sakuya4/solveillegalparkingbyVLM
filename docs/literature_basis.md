# Methodological Foundations

## Research Scope

This study connects temporal traffic-incident recognition to camera allocation
under a finite monitoring budget. The methodological basis comprises
peer-reviewed work in video analysis, edge intelligence, road-safety evaluation,
and combinatorial optimization. Original model papers and official runtime
documentation specify implementation details.

## Literature-to-Implementation Matrix

| Reference | Venue | Methodological implication | Repository implementation |
| --- | --- | --- | --- |
| Fang et al., 2024, [Vision-Based Traffic Accident Detection and Anticipation: A Survey](https://doi.org/10.1109/TCSVT.2023.3307655) | IEEE Transactions on Circuits and Systems for Video Technology | Distinguish detection from anticipation, use temporal endpoints, and examine geographic generalization | Causal window-end decisions, event recall, trigger delay, IID/geographic splits |
| Xu, Razavi, and Zheng, 2023, [Edge Video Analytics: A Survey on Applications, Systems and Enabling Techniques](https://doi.org/10.1109/COMST.2023.3323091) | IEEE Communications Surveys & Tutorials | Evaluate the interaction between model accuracy, compute allocation, communication, and latency | Candidate gating, ONNX execution, multi-camera queue replay, selective review |
| Arthurs et al., 2022, [A Taxonomy and Survey of Edge Cloud Computing for Intelligent Transportation Systems and Connected Vehicles](https://doi.org/10.1109/TITS.2021.3084396) | IEEE Transactions on Intelligent Transportation Systems | Place latency-sensitive inference near the camera and allocate heavier analysis to available resources | Edge TCN, asynchronous desktop worker, evidence review stage |
| Persaud and Lyon, 2007, [Empirical Bayes before-after safety studies: Lessons learned from two decades of experience and future directions](https://doi.org/10.1016/j.aap.2006.09.009) | Accident Analysis & Prevention | Address regression to the mean, traffic exposure, and temporal changes when estimating intervention effects | Temporal government-data holdout; a future controlled intervention protocol |
| Nemhauser, Wolsey, and Fisher, 1978, [An analysis of approximations for maximizing submodular set functions-I](https://doi.org/10.1007/BF01588971) | Mathematical Programming | Monotone nonnegative coverage permits a greedy cardinality-constrained approximation | Marginal-gain camera selection with matched-budget baselines |

The matrix links published methods to experimental design and executable
components. References were selected for methodological relevance and
peer-reviewed venue. Venue profiles identify the indexing database, metric
year, and subject category when reporting JCR or Scopus quartiles.

The [official IEEE CASS journal page](https://ieee-cas.org/publication/tcsvt)
documents TCSVT's scope across video acquisition, analysis, understanding, and
hardware/software implementation. The
[IEEE Communications Society journal directory](https://www.comsoc.org/publications/journals)
provides the publisher's journal information.

## Primary Implementation Sources

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition
Letters*, 27(8), 861--874. [DOI](https://doi.org/10.1016/j.patrec.2005.10.010).
This motivates separating ranking quality from an operating decision selected
under application costs. The continuous-event study adapts that principle to
episode budgets on held-out training clips; it does not equate event recall
with a window ROC coordinate.

Garwood, F. (1936). Fiducial Limits for the Poisson Distribution. *Biometrika*,
28(3--4), 437--442. [DOI](https://doi.org/10.1093/biomet/28.3-4.437).
The reviewed normal-exposure workflow uses the one-sided exact count bound,
including its zero-count limit, conditional on a stationary Poisson rate.
Source-clock coverage, label attribution and conflicting reviews are evaluated
before computing this interval; they are not resolved by the interval itself.

- [ACCIDENT benchmark](https://github.com/accidentbench/ACCIDENT): fixed-CCTV
  incident metadata and official data partitions.
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE): pretrained video encoder and
  fine-tuning design.
- [SAM 2](https://github.com/facebookresearch/sam2): prompted spatial evidence.
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL): structured multimodal review.
- [NAFNet](https://github.com/megvii-research/NAFNet): restoration baseline.
- [SM3Det](https://github.com/zcablii/SM3Det): multi-sensor detection architecture.
- [ONNX Runtime QNN execution provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html):
  target backend selection, quantized HTP models, and CPU-fallback control.
- [Qt QProcess](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QProcess.html):
  asynchronous native application jobs and process lifecycle management.
- [Reducto, SIGCOMM 2020](https://www.cs.princeton.edu/~ravian/publications/reducto_sigcomm20.pdf):
  early filtering and downstream resource efficiency in video analytics.
- [RFC 3290, Appendix A](https://www.rfc-editor.org/rfc/rfc3290.html#appendix-A):
  strict token-bucket conformance for measured-service review admission.
- [Accellera SystemC 3.0.2](https://github.com/accellera-official/systemc/releases/tag/3.0.2):
  transaction-level camera, inference, and evidence-queue simulation.
- [scikit-learn AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
  and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html):
  window-level ranking metrics in the matched representation study.
- [scikit-learn threshold selection](https://scikit-learn.org/stable/modules/classification_threshold.html):
  independent threshold tuning and separation of score estimation from decisions.

DOI titles for the five journal references were checked against Crossref during
this implementation. Source access date: 16 October 2026.
