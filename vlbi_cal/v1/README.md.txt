vlbi_cal v1.

This is the secon run of processing SMA's VLBI data to better understand the PID controller for the phasing loop.
We hope to improve the effeciency of the array during VLBI (EHT) observations.

Changes from v0:
- k_p, k_i, k_d ranges in SWARM_Opt scripts have changed to be narrower for "finer" resolution
	New ranges:
		k_p: 0.2, 1.5
		k_i: -0.15, 0.2
		k_d: -0.1, 0.5
- Instead of looking for the peak values per day, we are normalizing each day by the max effeciency.
- Each year has a "BAD" folder that contains data files that did not look right nor did not run correctly in v0.

File contents:
- Years of VLBI data.
- PLOTS : Plots and histograms of all years k_p, k_i, k_d values put up against each other.

Year folder contents: 
- vlbi_cal.###-#### ": VLBI observations for that year
- ####_SWARM_Opt : This python script computes the VLBI data to get the best calibration and effeciency values.
- ###-####_results_arr : .npz file with the resulting calibration values from SWARM_Opt script.
- ###_####_cal_solutions : .json file that contains the best k_p, k_i, k_d, and the max ("best") effeciency.