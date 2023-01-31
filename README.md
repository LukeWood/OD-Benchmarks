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

## Repo Structure

The repo is structured as follows:

- lib/ contains generic components, such as the logic to run augmentation, fit(), etc
- each experiment exists in a subdirectory of experiments/
- any configuration for experiments should live in this directory

## Results

Results are aggregated into a few files, this is still sort of a TODO
