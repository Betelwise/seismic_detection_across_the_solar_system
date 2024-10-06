# seismic_detection_across_the_solar_system

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
This project is an onboard seismic event detection system designed to optimize the way planetary landers, such as those on Mars, handle seismic data. Instead of transmitting large volumes of continuous seismic data back to Earth—which is costly, time-consuming, and energy-intensive—our algorithm processes the data directly onboard. It filters out noise, identifies seismic events, and only transmits the most relevant information to Earth.

Our system is a hybrid of traditional seismic analysis techniques and modern machine learning methods. By employing a Convolutional Neural Network (CNN), we ensure accurate event detection while maintaining computational efficiency.

## Features
- **Automatic detection of seismic events**
- **Efficient data processing**
- **Uses Convolutional Neural Networks (CNN)**
- **Processes a month of data in under 30 seconds**
- **Can work in high noise environment**
- **Tunnable to adapt different mission environments**

## How it Works
1. **Frequency Windowing:** The algorithm analyzes a full day of seismic data at once, automatically finding the best frequency window for isolating potential quakes.
2. **Outlier Removal:** High-amplitude, short-burst noise is removed to prevent interference with the subsequent processing steps.
3. **Normalization:** The data is normalized to ensure consistency in amplitude across different time periods.
4. **STA/LTA Analysis:** Short-term average/long-term average (STA/LTA) analysis is used to compare seismic events against background noise over a longer timeframe.
5. **Noise Filtering:** Excessive false signals are eliminated by analyzing how quickly the STA/LTA returns to its mean value.
6. **CNN Classification:** The final step involves running detected events through a CNN, which further refines the data by distinguishing true seismic events from false positives.
7. **Cataloging:** The identified seismic events are cataloged, ensuring only the most important data is transmitted back to Earth.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/betelwise/seismic_detection_across_the_solar_system
.git

2. Install the required dependencies, This was developed in packages defined in [requirements.txt](requirements.txt)txt