# Seismic detection across the Solar System

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How it Works](#How-it-Works)
- [Installation](#Installation)
- [How to test the results with your own Data](#How-to-test-the-results-with-your-own-Data)
- [How to see the results](#How-to-see-the-results)
- [Nasa SpaceApps Team Page](#Nasa-SpaceApps-Team-Page)
- [Contact](#Contact)
- [license](#license)

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


2. Install the required dependencies, This was developed in packages defined in [requirements](requirements.txt). These requirements are, if you want to train your own CNN or make some other changes, howver for making predictions or analyzing results all are not neccessary.
    ```bash
    pip install -r requirements.txt

## How to test the results with your own Data

If you want to make predicitons on your own dataset, following is the method to do so;
1. **Place your Data:** Place your miniseed files folder of Continuos Seismograms in [data](data).

2. **Enter your filePath:** Now open [predictions](predictions.ipynb) notebook and in 3rd cell comment out all foldersPath and place your own path to dataset. You may adjust the tuning parameters for better results, otherwise default tuning settings will be for apollo lunar seismic experiments.

3. **Specify maximum files to display** (optional): If you have a very large dataset. Making Plots of those files may take some time. You can specify how many max files to work on in 6th cell, default settings are 100 max files.

4. **Visualize the results:** in 2nd last cell you can see the results, it will plot all the files and detected events, it will show events that were filtered by conventional algorithms in vertical lines, all the events that were predicted to be positive signals will be represented in green lines and all the events predicted to be false signals will be in Red.

5. **Save the predicitons:** uncomment the save line in last cell and run this cell to save your predicitons in csv format.



## How to see the results


## Nasa SpaceApps Team Page
Find out more about the Challenge and Project in our SpaceApps Team Page
[Team Page Betelwise](https://www.spaceappschallenge.org/nasa-space-apps-2024/find-a-team/betelwise1/)

## Contact
For any questions, feel free to reach out to us via email at [Betelwise.com](betelwise.com).
