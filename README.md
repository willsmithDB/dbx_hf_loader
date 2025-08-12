# Databricks Huggingface Model Loader

A Python class for loading and deploying Hugging Face language models on Databricks Machine Learning Runtime clusters.

## Overview

This library provides utilities to efficiently load large language models from Hugging Face Hub into Databricks environments, with optimized caching and memory management for MLflow deployment.

## Features

- **Optimized Model Loading**: Efficient downloading and caching of Hugging Face models
- **Databricks Integration**: Seamless integration with Databricks ML Runtime and Unity Catalog volumes
- **MLflow Support**: Built-in MLflow model signature and deployment capabilities
- **Memory Management**: Support for quantization and optimized model loading
- **Volume Caching**: Persistent model caching using Databricks Unity Catalog volumes

## Requirements

This library is designed to work with Databricks Machine Learning Runtime. Install dependencies from `requirements.txt`:
