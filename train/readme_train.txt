Files:
- gsr_test.csv - Galvanic skin response sampled at 1Hz.
- hr_test.csv - Heart Rate sampled at 1Hz.
- rr_test.csv - RR intervals resampled at 1Hz.
- temp_test.csv - Skin temperature sampled at 1Hz.
- labels_train.csv - The first column contains the data labels (0 - cognitive load, 1-resting). The second column contains the user_id.

Each file contains 632 lines x 30 columns, 
corresponding to 632 instances each containing 
30 samples (30 seconds at the sampling rate 1 Hz). 
The train instances are randomly permuted.