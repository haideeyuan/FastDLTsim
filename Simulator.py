from utilis import init, output_workload, run_jobs, output_workload, output_finish, gen_metric
from scheduler import demo_scheduler

if __name__ == '__main__':
    Workloads, Err, scheduler, gpu_num = init("Env.json", "workload.csv", demo_scheduler)
    # Err = {1: [[324, 459], [180, 90]]}
    Err = {0: [[83, 936], [220, 110]], 2: [[83, 936], [220, 110]],3: [[83, 936], [220, 110]]}
    print("Err:", Err)
    finish_dict = run_jobs(Workloads, Err, scheduler, gpu_num, p_flag=True)
    print("finish_dict: ")
    output_finish(finish_dict)
    # gen_metric()