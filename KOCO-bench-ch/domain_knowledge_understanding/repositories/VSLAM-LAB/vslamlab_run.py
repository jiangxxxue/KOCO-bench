import argparse, os, yaml, time
from inputimeout import inputimeout, TimeoutOccurred
import pandas as pd

from Run.run_functions import run_sequence
from vslamlab_eval import evaluate, compare
from Datasets.get_dataset import get_dataset
from utilities import ws, show_time, filter_inputs
from Baselines.get_baseline import get_baseline
from vslamlab_utilities import check_experiments
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION, EXP_YAML_DEFAULT, VSLAM_LAB_DIR
import subprocess

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def main():

    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--exp_yaml', nargs='?', type=str,
                        const=EXP_YAML_DEFAULT, default=EXP_YAML_DEFAULT,
                        help=f"Path to the YAML file containing the list of experiments. "
                             f"Default \'vslamlab --exp_yaml {EXP_YAML_DEFAULT}\'")

    parser.add_argument('-run', action='store_true', help="")
    parser.add_argument('-evaluate', action='store_true', help="")
    parser.add_argument('-compare', action='store_true', help="")

    parser.add_argument('-ablation', action='store_true', help="")
    parser.add_argument('-overwrite', action='store_true', help="")

    args = parser.parse_args()
    args.exp_yaml = os.path.join(VSLAM_LAB_DIR, 'configs', args.exp_yaml)

    if not os.path.exists(VSLAMLAB_EVALUATION):
        os.makedirs(VSLAMLAB_EVALUATION, exist_ok=True)

    # Load experiment info
    experiments = check_experiments(args.exp_yaml, overwrite=args.overwrite)

    # Process experiments
    filter_inputs(args)

    if args.run:
        run(experiments, args.exp_yaml, ablation=args.ablation)

    # if args.evaluate:
    #     evaluate(experiments, overwrite=args.overwrite)

    # if args.compare:
    #     compare(experiments, args.exp_yaml)

def run(experiments, exp_yaml, ablation=False):
    print(f"\n{SCRIPT_LABEL}Running experiments (in {exp_yaml}) ...")
    start_time = time.time()

    completed_runs = {}
    not_completed_runs = {}
    num_executed_runs = 0
    duration_time_total = 0

    all_experiments_completed = {exp_name: False for exp_name in experiments}
    while True:
        if all(all_experiments_completed.values()):
            break

        remaining_iterations = 0
        for [exp_name, exp] in experiments.items():
            exp_log = pd.read_csv(exp.log_csv)
            completed_runs[exp_name] = (exp_log["STATUS"] == "completed").sum()  
            not_completed_runs[exp_name] = (exp_log["STATUS"] != "completed").sum() 
            remaining_iterations += not_completed_runs[exp_name]

            if not_completed_runs[exp_name] == 0:
                all_experiments_completed[exp_name] = True
                continue
                
            first_not_finished_experiment = exp_log[exp_log["STATUS"] != "completed"].index.min()
            row = exp_log.loc[first_not_finished_experiment]
            baseline = get_baseline(row['method_name'])
            dataset = get_dataset(row['dataset_name'], VSLAMLAB_BENCHMARK)    

            #process = subprocess.Popen("pixi run --frozen -e default clean_swap", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            #_, _ = process.communicate()
            results = run_sequence(row['exp_it'], exp, baseline, dataset, row['sequence_name'], ablation)

            duration_time = results['duration_time']
            duration_time_total += duration_time
            num_executed_runs += 1
            remaining_iterations -= 1

            exp_log["STATUS"] = exp_log["STATUS"].astype(str)
            exp_log["SUCCESS"] = exp_log["SUCCESS"].astype(str)
            exp_log["COMMENTS"] = exp_log["COMMENTS"].astype(str)
            exp_log.loc[first_not_finished_experiment, "STATUS"] = "completed"
            exp_log.loc[first_not_finished_experiment, "SUCCESS"] = results['success']
            exp_log.loc[first_not_finished_experiment, "COMMENTS"] = results['comments']
            exp_log.loc[first_not_finished_experiment, "TIME"] = duration_time
            exp_log.loc[first_not_finished_experiment, "RAM"] = results['ram']
            exp_log.loc[first_not_finished_experiment, "SWAP"] = results['swap']
            exp_log.loc[first_not_finished_experiment, "GPU"] = results['gpu']
            exp_log.to_csv(exp.log_csv, index=False)
                
            all_experiments_completed[exp_name] = exp_log['STATUS'].eq("completed").all()

        if(duration_time_total > 1):
            print(f"\n{SCRIPT_LABEL}: Experiment report: {exp_yaml}")
            print(f"{ws(4)}\033[93mNumber of executed iterations: {num_executed_runs} / {num_executed_runs + remaining_iterations} \033[0m")
            print(f"{ws(4)}\033[93mAverage time per iteration: {show_time(duration_time_total / num_executed_runs)}\033[0m")
            print(f"{ws(4)}\033[93mTotal time consumed: {show_time(duration_time_total)}\033[0m")
            print(f"{ws(4)}\033[93mRemaining time until completion: {show_time(remaining_iterations * duration_time_total / num_executed_runs)}\033[0m")

    run_time = (time.time() - start_time)
    print(f"\033[93m[Experiment runtime: {show_time(run_time)}]\033[0m")

if __name__ == "__main__":
    main()
