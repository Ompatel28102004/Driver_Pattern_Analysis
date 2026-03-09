## Overview

This repository contains a dataset and analysis code for studying **Traffic- and Road-Context Aware Aggressive Riding Detection for Two-Wheelers Using Smartphone Sensors**. The project collects motion and GPS data during real-world trips and analyzes it to understand driving patterns and road conditions.

The dataset includes multiple trips recorded using a mobile application that logs various onboard sensors such as accelerometer, gyroscope, and GPS. These recordings can be used to study driving dynamics, mobility patterns, and road surface events.

The repository provides:

* Raw sensor datasets collected from real trips
* Code for processing and analyzing the data
* Visualization outputs showing driving behavior along routes

---

## Repository Structure

```
Code-Dataset-Om
│
├── DataSet/
│   ├── DATASET_DESCRIPTION.md
│   ├── trip ZIP files containing raw sensor data
│
├── Output/
│   ├── generated interactive map visualizations
│
└── Driver_Pattern_Analysis.ipynb
    main notebook used for data processing and analysis
```

### DataSet

Contains the **raw sensor dataset** collected during multiple trips.
Each ZIP file corresponds to a **single trip recording** and includes smartphone sensor data in CSV format.

Detailed dataset documentation is available in:

```
DataSet/DATASET_DESCRIPTION.md
```

---

### Driver_Pattern_Analysis.ipynb

The main Jupyter Notebook used to run the project pipeline, including:

* loading and preprocessing sensor data
* analyzing driving patterns
* generating route-based visualizations

---

### Output

Contains generated outputs from the analysis such as:

* interactive driving behavior maps

These maps visualize route segments and detected events during the trip.

---

## Dataset Usage

The dataset provided in this repository is intended for **research and educational purposes**.

If you use this dataset or repository in research work, publications, or derivative projects, please provide appropriate acknowled
