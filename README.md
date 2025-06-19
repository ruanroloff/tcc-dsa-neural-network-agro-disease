# Neural networks for disease detection in agricultural crops

This project demonstrates how to use Deep Learning techniques to analyze images of crop plantations for automatic disease detection.

---

## Prerequisites

- AWS Account with appropriate permissions.
- Familiarity with Amazon SageMaker and Convolutional neural network (CNN).
- Download the datasets and upload to Amazon S3.


---

## Setup

Amazon SageMaker:
1. Ensure you have the necessary AWS permissions to run Amazon SagaeMaker notebooks and access to Amazon S3.
2. Clone this repository in your Amazon SagaeMaker Notebook instance.
3. Create a new Amazon S3 datalake.

---

## Project Organization
------------

    ├── LICENSE                 <- License file.
    ├── README.md               <- The top-level README for developers using this project.
    │
    ├── dataset
    │   └── sample              <- Data sample from public sources.
    │
    ├── docs                    <- Data dictionaries, manuals, pictures, and all other explanatory materials.
    │
    ├── model                   <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks               <- Jupyter notebooks using in the Exploratory phase.
    │
    ├── requirements.txt        <- The requirements file for reproducing the analysis environment, e.g.
    │                               generated with `pip freeze > requirements.txt`
    │
    ├── scripts                 <- Scripts to run in the notebook instance.
    │
    └── src                     <- Source code samples.


---

# Key Features

- Utilization of Deep Learning frameworks for image processing
- Analysis including:
  -  Healthy and unhealthy leaf images
  -  Model metrics

---

## Dependencies

- Tensorflow
- Keras
- Efficientnet

---

## Datasets

The PlantVillage dataset consists of 54303 healthy and unhealthy leaf images divided into 38 categories by species and disease.

- [The PlantVillage dataset](https://data.mendeley.com/datasets/tywbtsjrjv/1)

---

## References

This project used as reference the following research materials and experiments:

- [Amgad Shalaby](https://www.linkedin.com/in/amgad-shalaby-711287228/)
- [Shandong Agricultural University](http://english.sdau.edu.cn/)

---

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

---


## License

This library is licensed under the MIT-0 License. See the LICENSE file for more information.
