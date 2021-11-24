import os, sys, time, argparse
import numpy as np
import json
import ray

ray.init(log_to_driver=False)

def load_swarm_data(filename):
    with open(filename) as json_file:
        swarm_data = json.load(json_file)
    # Grab the total length of the data real quick
    n_data = len(swarm_data)

    # We're gonna be doing a lot of diff operations, which means in some cases we'll want
    # to pad some arrays with zeros. Construct some arrays now for the sake of convenience
    zero_pad = [
        np.zeros((1,len(swarm_data[0]['phases_lsb']))),
        np.zeros((1,len(swarm_data[0]['phases_usb']))),
    ]

    # These are the implemented phase values recorded in SWARM
    phases_lsb = np.array([swarm_data[idx]['phases_lsb'] for idx in range(n_data)])
    phases_usb = np.array([swarm_data[idx]['phases_usb'] for idx in range(n_data)])

    # These are the derived offsets/error terms for each antenna, given the implemented values
    cal_solution_lsb = np.array([swarm_data[idx]['cal_solution_lsb'][2] for idx in range(n_data)])
    cal_solution_usb = np.array([swarm_data[idx]['cal_solution_usb'][2] for idx in range(n_data)])

    # Let's calculate the "true" phase -- that is, assume that the solutions are perfect, and
    # use that to figure out what the antenna phase should _actually_ have been at time of obs.
    # There's kind of a funny padding operation that's needed here because of the order values
    # in the JSON file are recorded (soln's derived -> values implemented -> values recorded).
    true_phase_lsb = np.concatenate((zero_pad[0], phases_lsb[:-1] + cal_solution_lsb[1:]))
    true_phase_usb = np.concatenate((zero_pad[1], phases_usb[:-1] + cal_solution_usb[1:]))

    # Convert times from UNIX -> fractional UTC hours
    time_stamps = (np.array([swarm_data[idx]['int_time'] for idx in range(n_data)]) % 86400) / 3600.0
    phase_vals = np.concatenate((true_phase_lsb, true_phase_usb), axis=1).astype(np.float32)
    
    return (phase_vals, time_stamps)

@ray.remote
def sim_pid_loop(phase_data, int_length=8, kp=0.75, ki=0.40, kd=0.01):
    n_times = phase_data.shape[0]
    n_inputs = phase_data.shape[1]
    int_window = np.zeros((int_length, n_inputs), dtype=np.float32)
    new_epsilon = np.zeros((n_times, n_inputs), dtype=np.float32)
    last_cal = np.array(phase_data[0], dtype=np.float32)

    for idx in range(n_times):
        cal_soln = (((phase_data[idx] - last_cal) + 180.0 ) % 360.0) - 180.0
        new_epsilon[idx] = cal_soln
        #int_window = np.roll(int_window, 1, axis=0)
        #int_window[0] = cal_soln
        pos_mark = np.mod(idx, int_length)
        int_window[pos_mark] = cal_soln

        pid_response = (
            (kp * cal_soln)
            + (int_window.sum(axis=0) * (ki / int_length))
            + ((int_window[pos_mark] - int_window[pos_mark - 1]) * kd)
        )
        last_cal += pid_response
        last_cal = ((last_cal + 180.0 ) % 360.0) - 180.0

    ph_eff_vals = np.abs(
        np.mean(np.exp(-1j*np.deg2rad(new_epsilon.reshape((n_times, 8, -1)))),axis=2)
    )**2.0

    return ph_eff_vals

parser = argparse.ArgumentParser(description='Analyze a thing')
parser.add_argument('dataset', help='The data file to process')
parser.add_argument('output', help='Name of the output file')
args = parser.parse_args()

data_file = args.dataset
out_file = args.output 

n_kp = 21
kp_range = [0., 2.]
n_ki = 26
ki_range = [-10, 15]
n_kd = 21
kd_range = [-1., 1.]

n_int = 1
int_start = 3
int_step = 3

phase_arr, time_vals = load_swarm_data(data_file)
n_times = phase_arr.shape[0]
phase_arr_id = ray.put(phase_arr)
pid_arr = {}
print("Processing", end="")
sys.stdout.flush()

for idx, kp in enumerate(np.linspace(kp_range[0], kp_range[1], num=n_kp)):
    for jdx, ki in enumerate(np.linspace(ki_range[0], ki_range[1], num=n_ki)):
        for kdx, kd in enumerate(np.linspace(kd_range[0], kd_range[1], num=n_kd)):
            for ldx, int_length in enumerate(np.arange(int_start, int_start + (int_step * n_int) , int_step)):
                pid_arr[sim_pid_loop.remote(phase_arr_id, int_length=int_length, kp=kp, ki=ki, kd=kd)] = (idx,jdx,kdx,ldx)
    print(".", end="")
    sys.stdout.flush()
print("complete!")

results_arr = np.zeros((n_kp, n_ki, n_kd, n_int, n_times, 8), dtype=np.float32)
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

