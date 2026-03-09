## Overview

This dataset contains **raw smartphone sensor recordings collected during two-wheeler trips** for analyzing driving behavior and road surface conditions. The data is captured using a mobile application that logs multiple sensors simultaneously while the vehicle is in motion.

Each trip is stored as a **separate ZIP archive**, which contains individual CSV files corresponding to different sensors recorded by the smartphone.

The dataset is intended for research and experimentation in areas such as:

* Driver behavior analysis
* Aggressive vs non-aggressive driving detection
* Road anomaly detection (bumps and potholes)
* Traffic congestion analysis
* Smartphone sensor-based mobility analytics

---

# Dataset Structure

Each trip is stored as a compressed archive with the following format:

```
[prefix].zip
```

Example:

```
Normal_24-6-25_naroda-to-dau.zip
```

Inside each ZIP file, the dataset contains multiple CSV files representing different smartphone sensor streams.

Example structure:

```
Normal_24-6-25_naroda-to-dau.zip
│
├── Accelerometer.csv
├── Gyroscope.csv
├── Magnetometer.csv
├── Gravity.csv
├── Orientation.csv
└── Location.csv
```

---

# Sensor Data Files

### Accelerometer.csv

Records linear acceleration of the device along the three axes.

Sampling rate: **40 Hz**

Columns:

```
time | x | y | z
```

Unit: **m/s²**

---

### Gyroscope.csv

Measures angular velocity of the device.

Sampling rate: **40 Hz**

Columns:

```
time | x | y | z
```

Unit: **rad/s**

---

### Magnetometer.csv

Measures the Earth's magnetic field along three axes.

Sampling rate: **40 Hz**

Columns:

```
time | x | y | z
```

Unit: **µT**

---

### Gravity.csv

Represents the gravity component of acceleration detected by the device.

Sampling rate: **40 Hz**

Columns:

```
time | x | y | z
```

Unit: **m/s²**

---

### Orientation.csv

Provides the device orientation represented as quaternions.

Sampling rate: **40 Hz**

Columns:

```
time | qw | qx | qy | qz
```

These values describe the **3D rotation of the smartphone**, which is used to transform sensor readings into a vehicle-aligned coordinate frame.

---

### Location.csv

Contains GPS information recorded during the trip.

Sampling rate: **1 Hz**

Columns typically include:

```
time | latitude | longitude | speed | bearing | bearingAccuracy| horizontalAccuracy
```

Where:

* **latitude / longitude** → geographic coordinates
* **speed** → vehicle speed (m/s)
* **bearing** → direction of travel in degrees
* **horizontalAccuracy** → estimated GPS accuracy

---

# Sampling Frequencies

The dataset contains two types of sensor sampling rates:

| Sensor Type   | Sampling Rate |
| ------------- | ------------- |
| Accelerometer | 40 Hz         |
| Gyroscope     | 40 Hz         |
| Magnetometer  | 40 Hz         |
| Gravity       | 40 Hz         |
| Orientation   | 40 Hz         |
| GPS Location  | **1 Hz**      |

Motion sensors are recorded at **high frequency (40Hz)** to capture detailed driving dynamics, while GPS data is recorded at **1Hz** due to typical smartphone GPS limitations.

---

# Timestamp

All files include a column named:

```
time
```

This timestamp represents **nanoseconds since Unix epoch**.

During preprocessing, these timestamps are typically converted into human-readable datetime format.

---

# **Notes**

* Each ZIP file corresponds to **one driving trip**.
* Sensor files are stored **separately to preserve raw data integrity**.
* Data streams must be **synchronized and merged during preprocessing** for analysis.
---
