import os, sys, time, argparse
import matplotlib.pyplot as plt
import numpy as np
import json
import ray

ray.init(log_to_driver=False)
def load_swarm_data(filename):
    with open(filename) as json_file:
        swarm_data = json.load(json_file)
    # Grab the total number of integrations
    n_data = len(swarm_data)

    
    # 'cal_solution', 'delays', 'efficiencies', 'inputs', 'int_length', 'int_time', 'phases'
    if 'phases' in swarm_data[0].keys():
        # Phases is the "old" keyword, where DSB phasing wasn't used.
        input_count = np.median([len(swarm_data[idx]['inputs']) for idx in range(n_data)])
        use_data = [
            (len(swarm_data[idx]['inputs']) == input_count)
            and (len(swarm_data[idx]['phases']) == input_count)
            and (len(swarm_data[idx]['cal_solution'][2]) == input_count)
            for idx in range(n_data)
        ]

        swarm_data = [swarm_data[idx] for idx in range(n_data) if use_data[idx]]
        n_data = len(swarm_data)

        n_inputs = len(np.unique(np.array([data['inputs'] for data in swarm_data])[:, :, 0]))
        
        # These are the implemented phase values recorded in SWARM
        phase_online = np.array([swarm_data[idx]['phases'] for idx in range(n_data)])
        # These are the derived phase offsets post-correlation
        phase_solns = np.array([swarm_data[idx]['cal_solution'][2] for idx in range(n_data)])
    else:
        input_count = np.median([[len(data['inputs_lsb']), len(data['inputs_usb'])] for data in swarm_data])

        use_data = [
            (len(swarm_data[idx]['inputs_lsb']) == input_count)
            and (len(swarm_data[idx]['phases_lsb']) == input_count)
            and (len(swarm_data[idx]['cal_solution_lsb'][2]) == input_count)
            and (len(swarm_data[idx]['inputs_usb']) == input_count)
            and (len(swarm_data[idx]['phases_usb']) == input_count)
            and (len(swarm_data[idx]['cal_solution_usb'][2]) == input_count)
            for idx in range(n_data)
        ]

        swarm_data = [swarm_data[idx] for idx in range(n_data) if use_data[idx]]
        n_data = len(swarm_data)

        n_inputs = len(np.unique(np.array(
            [[data['inputs_lsb'], data['inputs_usb']] for data in swarm_data]
        )[:, :, :, 0]))

        # We're gonna be doing a lot of diff operations, which means in some cases we'll want
        # to pad some arrays with zeros. Construct some arrays now for the sake of convenience
        # These are the implemented phase values recorded in SWARM
        phase_online = np.concatenate(
            (
                np.array([data['phases_lsb'] for data in swarm_data]),
                np.array([data['phases_usb'] for data in swarm_data]),
            ),
            axis=1,
        )

        # These are the derived offsets/error terms for each antenna, given the implemented values
        phase_solns = np.concatenate(
            (
                np.array([data['cal_solution_lsb'][2] for data in swarm_data]),
                np.array([data['cal_solution_usb'][2] for data in swarm_data])
            ),
            axis=1,
        )

    # Let's calculate the "true" phase -- that is, assume that the solutions are perfect, and
    # use that to figure out what the antenna phase should _actually_ have been at time of obs.
    # There's kind of a funny padding operation that's needed here because of the order values
    # in the JSON file are recorded (soln's derived -> values implemented -> values recorded).
    # Add the two to get the "true" value at the time
    true_phases = phase_online[:-1] + phase_solns[1:]

        #true_phases = phases_usb[:-1] + cal_solution_usb[1:]
        #prog_vals = phases_usb

    # Convert times from UNIX -> fractional UTC hours
    time_stamps = (np.array([data['int_time'] for data in swarm_data]) % 86400) / 3600.0
    
    return (true_phases, n_inputs, time_stamps, phase_online)

def sim_pid_loop(phase_data, n_streams, int_length=8, kp=0.75, ki=0.40, kd=0.01):
    n_times = phase_data.shape[0]
    n_inputs = phase_data.shape[1]
    int_window = np.zeros((int_length, n_inputs))
    int_term = np.zeros(n_inputs)
    new_epsilon = np.zeros((n_times, n_inputs))
    pid_arr = np.zeros((n_times, n_inputs))
    last_cal = np.array(phase_data[0])
    last_epsilon = np.array(phase_data[0])

    for idx in range(n_times):
        cal_soln = (((phase_data[idx] - last_cal) + 180.0 ) % 360.0) - 180.0
        new_epsilon[idx] = cal_soln
        pos_mark = np.mod(idx, int_length)
        int_term += (cal_soln - int_window[pos_mark])
        int_window[pos_mark] = cal_soln
        del_term = cal_soln - last_epsilon

        pid_response = (
            (kp * cal_soln)
            + (int_term * (ki / int_length))
            + (del_term * kd)
        )
        last_cal += pid_response
        last_cal = ((last_cal + 180.0 ) % 360.0) - 180.0
        last_epsilon = cal_soln
        pid_arr[idx] = pid_response
    ph_eff_vals = (np.abs(
        np.mean(np.exp(-1j*np.deg2rad(new_epsilon.reshape((n_times, n_streams, -1)))),axis=2)
    )**2.0)
    return ph_eff_vals, pid_arr

@ray.remote
def get_pid_metrics(phase_data, n_streams, int_length=8, kp=0.75, ki=0.40, kd=0.01):
    ph_eff_vals, _ = sim_pid_loop(phase_data, n_streams, int_length=int_length, kp=kp, ki=ki, kd=kd)
    metric_arr = np.zeros(5)
    metric_arr[0] = np.mean(ph_eff_vals)
    metric_arr[1] = np.mean(ph_eff_vals**2)
    metric_arr[2] = np.mean(ph_eff_vals > 0.875)
    metric_arr[3] = np.mean(ph_eff_vals > 0.75)
    metric_arr[4] = np.mean(ph_eff_vals > 0.5)
    return metric_arr

parser = argparse.ArgumentParser(description='Analyze a thing')
parser.add_argument('dataset', help='The data file to process')
parser.add_argument('output', help='Name of the output file')
args = parser.parse_args()

data_file = args.dataset
out_file = args.output 

n_kp = 41
kp_range = [0., 2.]
n_ki = 51
ki_range = [-10, 15]
n_kd = 41
kd_range = [-1., 1.]

n_int = 12
int_start = 3
int_step = 1

phase_arr, n_inputs, time_vals, other_arr = load_swarm_data(data_file)
n_times = phase_arr.shape[0]
n_streams = phase_arr.shape[1] // n_inputs
phase_arr_id = ray.put(phase_arr)
pid_arr = {}
print("Processing", end="")
sys.stdout.flush()

for idx, kp in enumerate(np.linspace(kp_range[0], kp_range[1], num=n_kp)):
    for jdx, ki in enumerate(np.linspace(ki_range[0], ki_range[1], num=n_ki)):
        for kdx, kd in enumerate(np.linspace(kd_range[0], kd_range[1], num=n_kd)):
            for ldx, int_length in enumerate(np.arange(int_start, int_start + (int_step * n_int) , int_step)):
                pid_arr[get_pid_metrics.remote(
                    phase_arr_id, n_streams, int_length=int_length, kp=kp, ki=ki, kd=kd
                )] = (idx, jdx, kdx, ldx)
    print(".", end="")
    sys.stdout.flush()
print("complete!")

results_arr = np.zeros((n_kp, n_ki, n_kd, n_int, 5), dtype=np.float32)
print("Recording", end="")
sys.stdout.flush()
while pid_arr != {}:
    ready_ids, not_ready_ids = ray.wait(list(pid_arr.keys()), num_returns=n_ki*n_kd*n_int)
    for obj_id in ready_ids:
        results_arr[pid_arr[obj_id]] = ray.get(obj_id)
        del pid_arr[obj_id]
    print(".", end="")
    sys.stdout.flush()

np.save(out_file, results_arr)
print("complete!")

