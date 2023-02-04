# OD Benchmarking

KerasCV is a repo of modular CV components that allow computer vision researchers to
assemble state of the art computer vision pipelines --- but there are still open
questions:

- what pretrained backbone is optimal for OD on real world datasets
- what augmentation pipeline is optimal for training most OD models
- what scores for MaP and Recall can be achieved using KerasCV
- are anchor free models (YOLOx) as powerful as anchor based models

This repo attempts to answer all of these questions through extensive empirical
experimentation.

# Tasks:

Current tasks implemented:

- [PascalVOC](tasks/pascal_voc2007)
    - [Results](https://github.com/LukeWood/OD-Benchmarks/blob/master/tasks/pascal_voc2007/results/metrics.md)
- [Arthropods](tasks/arthropods) (results training now)
