vlbi_cal v0.

This is the first run of processing SMA's VLBI data to better understand the PID controller for the phasing loop.
We hope to improve the effeciency of the array during VLBI (EHT) observations.

File contents:
- Years of VLBI data.
- PLOTS : Plots and histograms of all years k_p, k_i, k_d values put up against each other.

Year folder contents: 
- vlbi_cal.###-#### ": VLBI observations for that year
- ####_SWARM_Opt : This python script computes the VLBI data to get the best calibration and effeciency values.
- ###-####_results_arr : .npz file with the resulting calibration values from SWARM_Opt script.
- ###_####_cal_solutions : .json file that contains the best k_p, k_i, k_d, and the max ("best") effeciency.