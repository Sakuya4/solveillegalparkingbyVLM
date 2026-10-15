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

The matrix identifies how published methods affect experimental design; it does
not transfer a source paper's benchmark performance to this system. Journal
quartiles depend on the indexing database, metric year, and subject category.
The references above were selected by topic and peer-reviewed venue. Any
submission-specific quartile table should record those three attributes using
the relevant JCR or Scopus edition.

The [official IEEE CASS journal page](https://ieee-cas.org/publication/tcsvt)
documents TCSVT's scope across video acquisition, analysis, understanding, and
hardware/software implementation. The
[IEEE Communications Society journal directory](https://www.comsoc.org/publications/journals)
provides the publisher's journal information.

## Primary Implementation Sources

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

DOI titles for the five journal references were checked against Crossref during
this implementation. Source access date: 16 October 2026.
