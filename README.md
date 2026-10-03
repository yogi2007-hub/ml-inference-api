# ML Inference API

A FastAPI service that returns class probabilities from a Random Forest classifier trained on scikit-learn's built-in Iris dataset.

> The assignment did not include a champion model. This repository trains a reproducible Iris demo model; it is not a supplied or externally validated champion model.

## API

`POST /predict`

Request:

```json
{"features": [5.1, 3.5, 1.4, 0.2]}
