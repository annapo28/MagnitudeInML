# Cardinality Augmented Loss Functions

Reproducibility implementation of **Cardinality Augmented Loss**, introduced in  
**"Cardinality Augmented Loss Functions"**.

---

## Preprint

**Cardinality Augmented Loss Functions**  
Miguel O'Malley  
arXiv preprint, 2026

```bibtex
@article{omalley2026cardinality,
  title   = {Cardinality Augmented Loss Functions},
  author  = {O'Malley, Miguel},
  journal = {arXiv preprint arXiv:2601.04941},
  year    = {2026}
}
```

---

## Overview

We propose **Cardinality Augmented Loss Functions**, implementations of cardinality-like invariants such as magnitude and the spread, in pursuit of improving results for training neural networks with class imbalance.

This repository contains code necessary to reproduce the experiments and results reported in the paper.

---

## Basic Usage

```bash
git clone https://github.com/miguelomalley/CardLoss.git
cd CardLoss
python imbalance.py
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

This work builds upon several existing tools and prior research:

* Our synthetic dataset generation is based on the work of Guyon et al. and utilizes the implementation provided by the `make_classification` function in *scikit-learn*

  > Guyon, I., Gunn, S., Nikravesh, M., & Zadeh, L. A. (2006). *Feature Extraction: Foundations and Applications*. Springer.
  > Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.

* Portions of this repository include a reproduction of the [**DeepGlassNet**](https://github.com/liubin06/DeepGlassNet) codebase, located in the `Glass/` directory, with modifications made solely for the purpose of integrating and evaluating the proposed loss functions.
  We take no credit for the original DeepGlassNet architecture or implementation beyond the specific loss-related modifications described in the paper.

```bibtex
@article{chen2024self,
      title={Self-Supervised Learning for Glass Composition Screening}, 
      author={Meijing Chen and Bin Liu and Ying Liu and Tianrui Li},
      journal = {Acta Materialia},
      volume = {301},
      pages = {121509},
      year = {2025},
      issn = {1359-6454},
      doi ={https://doi.org/10.1016/j.actamat.2025.121509}, 
}
```

