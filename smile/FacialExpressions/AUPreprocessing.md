### Methodology: Detailed Process of AU Signal Manipulation and Emotion Feature Computation

#### 1. **Input Data and Initial Preprocessing**
The input data comprises facial Action Units (AUs) extracted using the OpenFace tool. Each input file represents a session containing AU intensity values recorded over time, along with metadata such as timestamps and confidence scores.

##### **Libraries Used**: 
- `pandas` for data manipulation.
- `numpy` for numerical computations.
- `scipy.signal` for signal processing.


##### **Data Preparation**:
The preprocessing begins by reading the input CSV files into a pandas DataFrame. The relevant columns include the timestamps, frame numbers, success scores (indicating the confidence in facial recognition), and the AU intensity values.

- **Filtering by Confidence**:
  To ensure data quality, we filter out frames with low confidence. Specifically, if the 'success' column indicates a failed facial recognition, the corresponding AU values are set to zero. This step ensures that only reliable frames are considered for further analysis.

#### 2. **Standardization and Normalization**
- **Standardization**:
  The AU values are standardized by subtracting the mean and dividing by the standard deviation. This transformation centers the data around zero with a standard deviation of one, facilitating the comparison of AUs with different scales.

- **Normalization**:
  Normalization rescales the AU values to a range between 0 and 1 by subtracting the minimum value and dividing by the range (maximum - minimum). This process is particularly useful for ensuring that the features have comparable scales.

#### 3. **Signal Smoothing**
The AU signals often contain noise, which can be mitigated using a low-pass Butterworth filter. This filter is designed to pass signals with a frequency lower than a specified cutoff frequency while attenuating signals with higher frequencies.

- **Smoothing Process**:
  - **Sample Rate Calculation**: First, the sample rate is calculated based on the timestamps.
  - **Butterworth Filter Application**: A 3-pole low-pass Butterworth filter with a cutoff frequency set at 0.02 times the Nyquist frequency is applied to the AU signals using forward and backward filtering to ensure zero phase distortion.
  - **Adjustment for Non-Negative Values**: The filtered signal is then adjusted to ensure non-negative values, as negative intensities are not meaningful in this context.

#### 4. **Peak Detection**
To analyze facial expressions, significant peaks in the AU signals are identified. Peaks represent moments of high AU intensity, which are critical for detecting expressions.

- **Peak Detection Parameters**:
  - **Height**: Minimum height of the peaks.
  - **Prominence**: Minimum prominence of the peaks, which measures how much a peak stands out relative to its surroundings.
  - **Distance** and **Width**: Constraints on the minimum distance and width between peaks to avoid detecting noise as peaks.

#### 5. **Feature Computation for Single AUs**
For each AU, we compute several features based on the detected peaks. These features include:

- **Intensity (int_avg_combo)**: The mean intensity of the peaks.
- **Prominence (prominence_combo)**: The maximum prominence of the peaks.
- **Area (area_combo)**: The sum of the signal values within the peaks.
- **Duration (duration_combo)**: The duration of the peaks in seconds.
- **Low Peak Positions (peak_lp_combo)**: The positions of the low peaks.

These features are essential for characterizing the behavior of each AU and identifying specific expressions.

#### 6. **Feature Computation for Combined AUs**
Facial expressions are often better represented by combinations of multiple AUs. Thus, we also compute features for combinations of AUs.

- **Combination Feature Computation**:
  For each pair or group of AUs, we calculate combined features by considering the peaks across multiple AUs. This involves averaging the features of individual AUs to obtain a single set of features that represent the combined behavior of the AUs.

- **Average Combination Feature Computation**:
  - **Average Prominence**: The mean prominence of the peaks across the combined AUs.
  - **Average Intensity**: The mean intensity of the peaks across the combined AUs.
  - **Average Area**: The mean area under the peaks across the combined AUs.
  - **Average Duration**: The mean duration of the peaks across the combined AUs.

This averaging process yields a robust representation of facial expressions, capturing the synergistic effects of multiple AUs.

#### 7. **Emotion-Specific AU Combinations**
The script processes different emotions by extracting Action Units (AUs) that correspond to various expressions. Each emotion is analyzed in two categories: simple and combo expressions.

- **Simple Expressions**: Involve individual AUs that are typically associated with a specific emotion.
- **Combo Expressions**: Involve combinations of multiple AUs that together represent a more complex emotional expression.

The specific AUs used for each emotion in both categories are as follows:

- **Smiling**:
  - **Simple**: AU12 (lip corner puller)
  - **Combo**: AU12 (lip corner puller) and AU6 (cheek raiser)

- **Sadness**:
  - **Simple**: AU1 (inner brow raiser) and AU4 (brow lowerer)
  - **Combo**: AU1 (inner brow raiser), AU4 (brow lowerer), and AU15 (lip corner depressor)

- **Anger**:
  - **Simple**: AU4 (brow lowerer), AU5 (upper lid raiser), and AU7 (lid tightener)
  - **Combo**: AU4 (brow lowerer), AU5 (upper lid raiser), AU7 (lid tightener), and AU23 (lip tightener)

- **Fear**:
  - **Simple**: AU1 (inner brow raiser), AU2 (outer brow raiser), and AU4 (brow lowerer)
  - **Combo**: AU1 (inner brow raiser), AU2 (outer brow raiser), AU4 (brow lowerer), and AU5 (upper lid raiser)

For each emotion, the corresponding AUs are processed to compute both individual and combined features. These features are then saved to CSV files for further analysis.

#### 8. **Saving the Results**
The computed features for both individual and combined AUs are saved to CSV files. The files are organized by emotion and type (simple or combo), ensuring a structured dataset for downstream analysis.

