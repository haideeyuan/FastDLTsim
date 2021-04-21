import csv
import json
import numpy as np
import warnings
import random
import copy
import math


def init(env_file, workload_file, sched_policy):
    '''
    :param env_file: A json file, eg. ./Env.json
    :param workload_file: A csv file, eg. ./workload.csv
    :param sched_policy: A class
    :return:
        workloads: A nested dict. dict1{dict2} workloads[job_id] = dict2{}.
            The content of workloads[job_id] is a row of workload_file represents one job's information.
            The keys of dict2 is the first line in workload_file.
        err: A dict. err[gpu_id] = list([list1],[list2]).
            Use gpu_id as a key to index error time of the gpu.
            The value of err is a nested list, where the first list vik. list1 represents error start time, and the list2 represents error duration respectively.
    '''
    workloads = read_workload(workload_file)
    check_first_job_in_workload(workloads)
    # output_workload(workloads)
    env = read_env(env_file)
    env_para = env["Env_para"]
    scheduler = sched_policy()
    host_num = env["Env_para"]["Host_num"]
    gpu_num_per_host = env["Env_para"]["GPU_num_per_host"]
    gpu_num = host_num * gpu_num_per_host
    err = gen_error(GPU_num=gpu_num, High_err_num=env_para["High_error_card"],
                    High_err_rate=env_para["High_error_rate"],
                    Low_err_num=env_para["Low_error_card"], Low_err_rate=env_para["Low_error_rate"],
                    Max_err_num=env_para["Max_err_num"], Workload=workloads,
                    scheduler=scheduler)
    return workloads, err, scheduler, gpu_num


def check_first_job_in_workload(workload):
    job_id = list(workload.keys())[0]
    if workload[job_id]["submit_time"] != 0:
        raise ValueError("The submit time of first job in workload file (default is workload.csv) MUST be 0 (ZERO) second, %d second is NOT acceptable!" % workload[job_id]["submit_time"])


def read_workload(workload_file):
    '''
    return: dict[dict]. Each [dict] in dict represents a workload.(One raw in workload file).
    '''
    workloads = {}
    csv_reader = csv.reader(open(workload_file))
    i = 0
    for row in csv_reader:
        if i == 0:
            dic_key = row
            len_key = len(dic_key)
        if i > 0:
            w_dict = {}
            for j in range(len_key):
                if dic_key[j] in ["submit_time", "running_time", "GPU_num", "restart", "real_sub", "real_running",
                                  "preempt_times", "err_times", "place_times"]:
                    w_dict[dic_key[j]] = int(row[j])
                elif dic_key[j] in ["running_state", "finish_flag"]:
                    w_dict[dic_key[j]] = row[j] == str(True)
                elif dic_key[j] in ["decay", "score"]:
                    w_dict[dic_key[j]] = float(row[j])
                else:
                    w_dict[dic_key[j]] = row[j]
            if row[0] in workloads:
                raise NameError("Job ID [%s] repeated." % row[0])
            workloads[row[0]] = w_dict
        i = i + 1
    return workloads


def read_env(env_file):
    with open(env_file, 'r') as f:
        env = json.load(f)
    if env["Env_para"]["GPU_num_per_host"] * env["Env_para"]["Host_num"] < (
            env["Env_para"]["High_error_card"] + env["Env_para"]["Low_error_card"]):
        raise ValueError("The sum of 'Low_error_card' and 'High_error_card' is over than 'Total_GPU_NUM'.")
    return env


def get_GPU_req_num(Workloads, job_list):
    job_num = len(job_list)
    job_id = job_list.keys()
    GPU_req_num = 0
    for i in range(job_num):
        GPU_req_num += Workloads[job_id[i]]["GPU_num"]
    if GPU_req_num == 0:
        print("[Warining]: The request number of GPUs is ZERO.")
    return GPU_req_num


def get_sub_plus_run_time(Workloads):
    time = list()
    for i in Workloads.keys():
        time.append(int(Workloads[i]["submit_time"] + Workloads[i]["running_time"]))
    return time


def get_ideal_time(Workloads):
    # 1. GPU/CPU: No limited.    2. No error rate.
    # ideal_time = max(submit_time + runing_time)
    return max(get_sub_plus_run_time(Workloads))


def submitted_flag(Workload, job_list):
    for i in job_list:
        if "submitted" in Workload[i] and Workload[i]["submitted"]:
            return True
        else:
            return False


def unsubmit_job(Workload, job_list):
    unsub_job = list()
    for i in job_list:
        if "submitted" not in Workload[i]:
            unsub_job.append(i)
    return unsub_job


def modify_sub_time(Workload, mod_list, t):
    for i in mod_list:
        Workload[i]["submit_time"] = t


def get_sub_time(Workload, job_list):
    key_time = list()
    for i in job_list:
        key_time.append(Workload[i]["submit_time"])
    return key_time


def get_finish_order(Worload, job_list):
    time = list()
    for i in job_list:
        time.append(Worload[i]["submit_time"] + Worload[i]["running_time"])
    return np.argmin(time), np.min(time)


def gpu_limit_time(Workload, scheduler, gpu_rest_num):
    '''
    return: gpu不出错的时候最大完成时间JCT（资源利用率），平均等待时间
    '''
    finish = run_jobs(Workload, {}, scheduler, gpu_rest_num, p_flag=False)
    time = list()
    for job_id in finish.keys():
        time.append(finish[job_id][-1][1])
    return max(time), None


def gen_int_point(up_num, num):
    '''
    生成随机正整数点，生成范围是0~up_num，共生成num个【不重复】正整数，返回从小到大排序的列表。
    可以用于生成【有错误的卡号】、【出错的时间点】。
    '''
    l = [random.randint(0, up_num) for i in range(num)]
    l = list(set(l))
    l.sort()
    return l


def gen_int_point_expt(up_num, num, expt):
    l = list()
    for i in range(num):
        n = random.randint(0, up_num)
        while n in expt:
            n = random.randint(0, up_num)
        l.append(n)
    l = list(set(l))
    return l


def gen_time_period(maxValue, num):
    '''生成总和固定的随机整数序列，大小也是随机排序的。每一个值代表一个时间段。
    maxvalue: 序列总和
    num：要生成的整数个数'''
    a = random.sample(range(1, maxValue), k=num - 1)  # 随机生成num个数据
    a.append(0)  # 加上数据开头
    a.append(maxValue)
    a = sorted(a)
    b = [a[i] - a[i - 1] for i in range(1, len(a))]  # 列表推导式，计算列表中每两个数之间的间隔
    # print(sum(b))
    return b

def new_gen_time_period(refTime):
    return refTime


def gen_error(GPU_num, High_err_num, High_err_rate, Low_err_num, Low_err_rate, Max_err_num, Workload, scheduler, recover_time = 10, scale = 60):
    print("Generate error situation in a new way...\\")
    #生成出错卡id
    High_err_cards = gen_int_point(GPU_num-1, High_err_num)
    Low_err_cards = gen_int_point_expt(GPU_num-1, Low_err_num, High_err_cards)
    err_cards = High_err_cards + Low_err_cards
    #生成出错时间点
    err_dict = {}
    for i in range(len(err_cards)):
        err_dict[err_cards[i]] = [[],[]]
    T_org, _ = gpu_limit_time(copy.deepcopy(Workload), scheduler, GPU_num)
    print("[Predictor]: The predicted total running time of all workloads is %ds." % T_org)

    #scale = 60 #可调整参数，当前表示每60个时刻出错概率为err_rate
    lamda_high = -scale / math.log(1-High_err_rate)
    lamda_low = -scale / math.log(1-Low_err_rate)
    #生成高错误卡的出错分布
    for i in range(len(High_err_cards)):
        timer = 0
        while True:
            timer = timer + int(np.random.exponential(lamda_high,1))
            if (timer > T_org):
                break
            err_dict[err_cards[i]][0].append(timer)
            period = new_gen_time_period(recover_time)
            err_dict[err_cards[i]][1].append(period)
            timer = timer + period
    #生成低错误卡的出错分布
    for i in range(len(High_err_cards),len(err_cards)):
        timer = 0
        while True:
            timer = timer + int(np.random.exponential(lamda_low,1))
            if (timer > T_org):
                break
            err_dict[err_cards[i]][0].append(timer)
            period = new_gen_time_period(recover_time)
            err_dict[err_cards[i]][1].append(period)
            timer = timer + period
    return err_dict


def gen_error_old(GPU_num, High_err_num, High_err_rate, Low_err_num, Low_err_rate, Max_err_num, Workload, job_list):
    print("Generating error situation...\\")
    # 生成出错卡的id
    High_err_cards = gen_int_point(GPU_num, High_err_num)
    # print("H_card:", High_err_cards)
    Low_err_cards = gen_int_point_expt(GPU_num, Low_err_num, High_err_cards)
    # print("L_card:", Low_err_cards)
    err_cards = High_err_cards + Low_err_cards
    # print("err_card", err_cards)
    # 生成出错时间段
    err_num = random.randint(0, int(Max_err_num / 2))
    err_num = 1 if err_num == 0 else err_num
    T_org, _ = gpu_limit_time(copy.deepcopy(Workload), job_list, GPU_num)
    print("[Predictor]: The predicted total running time of all workloads is %ds." % T_org)
    High_time_point = gen_int_point(T_org, err_num)
    High_time_fix = int(High_err_rate * T_org)
    High_time_period = gen_time_period(High_time_fix, err_num)
    Low_time_point = gen_int_point(T_org, err_num)
    Low_time_fix = int(Low_err_rate * T_org)
    Low_time_period = gen_time_period(Low_time_fix, err_num)
    time_list = [High_time_point + Low_time_point, High_time_period + Low_time_period]
    print("time_list", time_list)
    err_dict = {}
    for i in range(len(err_cards)):
        err_dict[err_cards[i]] = time_list
    return err_dict


def cal_err_time(Workload, jobs, gpu_rest_num, err_dict):
    '''
    return: 最大完成时间JCT（资源利用率），平均等待时间
    '''
    t = 0  # JCT 最大完成时间
    wait_time = 0  # 等待时间总和
    gpu_org = gpu_rest_num
    job_list = list(jobs).deepcopy()
    for i, j in enumerate(job_list):
        if i == 0:
            if gpu_rest_num < Workload[j]["GPU_num"]:
                warnings.warn("Inital GPU number is too LOW to start the first job.")
                while True: pass
            else:
                t = Workload[j]["submit_time"] + Workload[j]["running_time"]
                # print("The value of t after first job is: ", t)
        else:
            # 计算默认submit_time时，gpu资源剩余量，并保留作业id即f_job_list供后面使用：
            f_job_list = list()
            for k in range(i):
                if Workload[j]["submit_time"] < t:
                    if Workload[j]["submit_time"] <= \
                            (Workload[job_list[k]]["submit_time"] + Workload[job_list[k]]["submit_time"]):
                        gpu_rest_num -= Workload[job_list[k]]["GPU_num"]
                        f_job_list.append(job_list[k])
            # print("The rest GPU resource of job %s: %d" % (j, gpu_rest_num))
            # print("The farword job list is: ", f_job_list)
            # 如果GPU资源剩余充足[再看是不是任务执行过程中GPU没有错误]，计算总用时t
            if gpu_rest_num >= Workload[j]["GPU_num"]:
                t_tmp = Workload[j]["submit_time"] + Workload[j]["running_time"]
                t = max(t, t_tmp)
            # 如果GPU资源不足，计算出j作业的【实际submit时间】，再得到总用时t：
            else:
                org_sub = Workload[j]["submit_time"]
                while True:
                    f_job, f_t = get_finish_order(Workload, f_job_list[:i])
                    gpu_rest_num += Workload[f_job_list[f_job]]["GPU_num"]
                    del f_job_list[f_job]
                    if gpu_rest_num >= Workload[j]["GPU_num"]:
                        Workload[j]["submit_time"] = f_t
                        print("Modified submitting time from %d to %d of job %s" % (org_sub, f_t, j))
                        wait_time += (f_t - org_sub)
                        t_tmp = Workload[j]["submit_time"] + Workload[j]["running_time"]
                        t = max(t, t_tmp)
                        break
    return t, wait_time / len(jobs)


def host2ext(host_id, gpu_id, gpu_per_host):
    gpus = (np.array(host_id) * gpu_per_host + np.array(gpu_id)).tolist()
    return gpus


def err_exist(err, job_id, workload, host_ids, gpu_ids, gpu_per_host):
    print("[Info]: Detecting errors of job %s." % job_id)
    job_start = workload[job_id]["real_sub"]
    job_finish = workload[job_id]["real_sub"] + workload[job_id]["real_running"]
    gpu_ids_ext = host2ext(host_ids, gpu_ids, gpu_per_host)
    for gpu_id in gpu_ids_ext:
        if gpu_id in err.keys():
            err_point = err[gpu_id][0]
            err_period = err[gpu_id][1]
            for k, t in enumerate(err_point):
                if job_start < t < job_finish:  # err occur
                    workload[job_id]["err_times"] += 1
                    print("[Info]: Detected valid error point at %s, last %s." % (err_point, err_period))
                    return t, t - job_start  # t: time when err occur.  t-job_start: job ran time
                elif job_start == t:
                    return -1, None
                else:
                    return None, None
    return None, None


def get_unfinish_job_num(workloads):
    num = 0
    for job_id in workloads:
        if not workloads[job_id]["finish_flag"]:
            num += 1
    return num


def get_metric(workload, host_num, err, scheduler, gpu_per_host, gpu_state):
    t = 0
    s = 10
    gpu_num = host_num * gpu_per_host
    unfinish_num = get_unfinish_job_num(workload)
    while unfinish_num != 0:
        # while s:
        #     s -= 1
        print("[Info]: %d job(s) left... %s gpu free... t=%d." % (
            unfinish_num, scheduler.get_free_gpu_id(t, gpu_state, gpu_num, err), t))
        job_list = scheduler.order()
        # print("[Simulator]: Order list is: ", job_list)
        for job_id in job_list:
            if not workload[job_id]["finish_flag"]:
                host_id, gpu_id = scheduler.place(job_id=job_id, gpu_num=gpu_num, gpu_num_per_host=gpu_per_host,
                                                  gpu_state=gpu_state, err=err)
                print("[Placer]: Job-%s should be located in host-%s and device-%s" % (job_id, host_id, gpu_id))
                if host_id is None:
                    print("[Info]: Modify the real_sub time of job %s from %d to %d" % (
                        job_id, workload[job_id]["real_sub"], max(t, workload[job_id]["real_sub"])))
                    workload[job_id]["real_sub"] = max(t, workload[job_id]["real_sub"])
                    break
                else:
                    e_time, e_ran = err_exist(err, job_id, workload, host_id, gpu_id, gpu_per_host)
                    p_time, p_ran = scheduler.preempt(job_id)
                    if e_time is None and p_time is None:  # Run job
                        workload[job_id]["gpus"] = host2ext(host_id, gpu_id, gpu_per_host)
                        # if t > workload[job_id]["real_sub"]:
                        # print("[Info]: Modify the real_sub time from %d to %d" % (workload[job_id]["real_sub"], t))
                        # workload[job_id]["real_sub"] = max(workload[job_id]["real_sub"], t)
                        t = workload[job_id]["real_sub"] + workload[job_id]["running_time"]
                        print("[Simulator]: Ran job %s, t --> %d" % (job_id, t))
                        gpu_state.append(
                            [workload[job_id]["real_sub"], t, host2ext(host_id, gpu_id, gpu_per_host), job_id])
                        workload[job_id]["finish_flag"] = True
                        continue
                    elif e_time == -1:
                        # t += 1
                        print("[Info]: Error occurred when schedule. Modify the real_sub time from %d to %d" % (
                            workload[job_id]["real_sub"], t))
                        workload[job_id]["real_sub"] = t
                        workload[job_id]["err_times"] += 1
                        workload[job_id]["restart"] += 1
                        continue
                    else:  # err occur
                        if p_time is None or e_time < p_time:  # err occur
                            print("[Info]: Error occur in job-%s" % job_id)
                            workload[job_id]["err_times"] += 1
                            if scheduler.dynamic():
                                workload[job_id]["real_running"] -= e_ran
                            workload[job_id]["restart"] += 1
                            workload[job_id]["real_sub"] = max(workload[job_id]["real_sub"], t)
                            t = e_time
                            print("[Simulator]: Error occurred in job %s, t --> %d" % (job_id, t))
                            gpu_state.append(
                                [workload[job_id]["real_sub"], t, host2ext(host_id, gpu_id, gpu_per_host), job_id])
                            break
                        else:  # preempt occur
                            print("[Info]: Preempt occur in job-%s" % job_id)
                            workload[job_id]["preempt_times"] += 1
                            if scheduler.dynamic():
                                workload[job_id]["real_running"] -= p_ran
                            workload[job_id]["restart"] += 1
                            workload[job_id]["real_sub"] = max(workload[job_id]["real_sub"], t)
                            t = p_time
                            print("[Simulator]: Preempt occurred in job %s, t --> %d" % (job_id, t))
                            gpu_state.append(
                                [workload[job_id]["real_sub"], t, host2ext(host_id, gpu_id, gpu_per_host), job_id])
                            break
        unfinish_num = get_unfinish_job_num(workload)
    print("gpu_state:", gpu_state)


def output_workload(workload):
    print("The workloads are: ")
    for job_id in workload.keys():
        print('\t', workload[job_id])


def output_finish(finish):
    print("The original running data: ")
    for job_id in finish.keys():
        print('\t', job_id, ': ', finish[job_id])


'''
**********************************************************************************************************************
2020.1.13 3.00 P.M.
**********************************************************************************************************************
'''


def get_first_job_in_workloads(Workloads):
    """
    Get the job_id with min(sub_time) and max(score)
    :return: A str. Represents a job_id.
    """
    job_id = list(Workloads.keys())[0]
    same_time = list()
    for _job_id in Workloads.keys():
        if Workloads[_job_id]["submit_time"] < Workloads[job_id]["submit_time"]:
            job_id = _job_id
    for _job_id in Workloads.keys():
        if Workloads[_job_id]["submit_time"] == Workloads[job_id]["submit_time"]:
            same_time.append(_job_id)
    if len(same_time) > 1:
        for _job_id in same_time:
            if Workloads[_job_id]["score"] > Workloads[job_id]["score"]:
                job_id = _job_id
    return job_id


def get_first_job_time_in_workloads(Workloads):
    """
    Get the next(first) job's submit time in workloads.
        >> first_job's submit time.
        >> Depend on function:
            >>> get_first_job_in_workloads()
    :return: A int time point.
    """
    job_id = get_first_job_in_workloads(Workloads)
    return Workloads[job_id]["submit_time"]


def get_last_sub_time(finish_dict):
    """
    Get the last job's submit time in finish_dict.
        >> Max submit time in finish_dict.
    :param finish_dict:
    :return: A int time point.
    """
    if len(finish_dict) == 0:
        return 0
    time = finish_dict[list(finish_dict.keys())[0]][0][0]
    for job_id in finish_dict.keys():
        l = len(finish_dict[job_id])
        for k in range(l):
            time = finish_dict[job_id][k][0] if finish_dict[job_id][k][0] > time else time
    return time


def gpu_increase_in_period(time1, time2, Err, finish_dict):
    """
    Is there any time point to rise gpu resource during time1 to time2, consider two conditions:
        1. Err --> Did error gpu(s) recover during time1~time2?
        3. finish_dict (pre jobs) --> Did pre jobs finished to release gpu resource?
    :param time1:
    :param time2:
    :param Err:
    :param finish_dict:
    :return: A list content increase time point if there is increasing.
            None if no gpu increase.
    """
    time = list()
    rise_time = list()
    for gpu_id in Err.keys():
        t = (np.array(Err[gpu_id][0]) + np.array(Err[gpu_id][1])).tolist()
        time += t
    for job_id in finish_dict.keys():
        info = finish_dict[job_id]
        for k in range(len(info)):
            t = info[k][1]
            time.append(t)
    for i in range(len(time)):
        if time1 <= time[i] < time2:
            rise_time.append(time[i] + 1)
    #
    rise_time.append(time1)
    rise_time.sort()
    return rise_time if rise_time != [] else None


def get_able_job_in_wait_q(time_points, wait_q, finish_dict, Err, gpu_num):
    """
    Get a job in wait_q which can be submitted.
    Conditions of submitting:
        1. Consider that if job has been interrupted before, and interruption time < time_point, then job can be submitted.
        2. Is there enough gpu resource when [time == time_points[i]].
    If there are more than one job can be submitted, then return the max score one.
    :param time_points:
    :param wait_q:
    :return: A str for job_id; A int for job starting time_point.
            None,None if no job can be submitted.
    """
    for time in time_points:
        able_jobs = list()
        free_gpu_num = len(get_free_gpu(time, finish_dict, Err, gpu_num))
        for job_id in wait_q.keys():
            if job_id in finish_dict.keys():
                if finish_dict[job_id][-1][-1] != "None" and finish_dict[job_id][-1][1] < time:
                    if wait_q[job_id]["GPU_num"] <= free_gpu_num:
                        able_jobs.append(job_id)
            else:
                if wait_q[job_id]["GPU_num"] <= free_gpu_num:
                    able_jobs.append(job_id)
        # print("debug:able_id=", able_jobs)
        if len(able_jobs) > 1:
            sub_job_id = able_jobs[0]
            for job_id in able_jobs:
                if wait_q[sub_job_id]["score"] < wait_q[job_id]["score"]:
                    sub_job_id = job_id
            return sub_job_id, time
        elif len(able_jobs) == 1:
            sub_job_id = able_jobs[0]
            return sub_job_id, time
        else:
            continue
    return None, None


def get_free_gpu(time, finish_dict, Err, gpu_num):
    """
    Get all free gpu(s) in time point. Consider about two situations:
        1. finish_dict --> The pre jobs occupy at time point.
        2. Err --> The gpu error at time point.
    :param time:
    :param finish_dict:
    :param Err:
    :return: Free gpu list.
        [] Empty list will be returned if there is no free gpu resource.
    """
    gpu_working_state = [False] * gpu_num
    for job_id in finish_dict.keys():
        for k in range(len(finish_dict[job_id])):
            if finish_dict[job_id][k][0] <= time <= finish_dict[job_id][k][1]:
                for i in range(len(finish_dict[job_id][k][2])):
                    gpu_working_state[finish_dict[job_id][k][2][i]] = True
    for gpu_id in Err.keys():
        n = len(Err[gpu_id][0])
        for i in range(n):
            if Err[gpu_id][0][i] <= time <= Err[gpu_id][0][i] + Err[gpu_id][1][i]:
                gpu_working_state[gpu_id] = True
    return [idx for idx, x in enumerate(gpu_working_state) if x is False]


# def get_err_or_preempt_flag(job_info, gpu_ids, Err, workloads):
#     """
#     To check that the job will be interrupted by gpu err, or later job preempt, or no interruption.
#     :param job_info: dict. wait_q or finish_info.
#     :param Err: dict{key: list([],[])}
#     :param workloads:
#     :return: str. "err", "preempt", None.
#     """


def get_err_or_preempt(job_info, gpu_ids, Err, workloads, preempt_flag):
    """
    Function 1:
        To check that the job will be interrupted by gpu err, or later job preempt, or no interruption.
    Function 2:
        Get the interrupt time point. Consider two situations:
            1. Error occur.
            2. Preempt occur.
                2.1 Later job in workload.
                2.2 [Note]: There is NO need to consider about wait_q.
    :param gpu_ids:
    :param job_info: wait_q or finish_info.
    :param Err:
    :param workloads:
    :return: flag, time, preempt_id
        flag: To denotes that the job will be interrupted by [gpu err], or [later job preempt], or [no interruption].
            [gpu err] -- "err";
            [later job preempt] -- "preempt"
            [no interruption] -- None
        time: An int represents interruption time point. If no interruption, time = None.
        preempt_id: If preemption occurred, return preempt job id. Preempt job will be submit next.
            If there is no preemption occur, return None.
    """
    # Err:
    err_times = list()
    for gpu_id in Err.keys():
        if gpu_id not in gpu_ids:
            continue
        for i in range(len(Err[gpu_id][0])):
            if job_info["submit_time"] < Err[gpu_id][0][i] <= job_info["submit_time"] + job_info["running_time"]:
                err_times.append(Err[gpu_id][0][i])
    err_time = None if err_times == [] else min(err_times)-1
    # Preempt:
    if preempt_flag:
        preempt_times = list()
        preempt_ids = list()
        t1 = job_info["submit_time"]
        t2 = t1 + job_info["running_time"]
        for job_id in workloads.keys():
            if t1 < workloads[job_id]["submit_time"] < t2 and workloads[job_id]["score"] > job_info["score"]:
                preempt_times.append(workloads[job_id]["submit_time"])
                preempt_ids.append(job_id)
        preempt_time = min(preempt_times) - 1 if preempt_times != [] else None
        preempt_id = preempt_ids[np.argmin(np.array(preempt_time))] if preempt_times != [] else None
    else:
        preempt_time = None
        preempt_id = None
    # Integration:
    if err_time is None and preempt_time is None:
        return None, None, None
    elif err_time is None:
        # print("no err but preempt exist")
        return "preempt", preempt_time, preempt_id
    elif preempt_time is None:
        return "err", err_time, None
    else:
        if err_time <= preempt_time:
            return "err", err_time, None
        else:
            return "preempt", preempt_time, preempt_id


def next_gpu_increase_time(time1, Err, finish_dict):
    """
    Is there any time point to rise gpu resource during time1 to time2, consider two conditions:
        1. Err --> Did error gpu(s) recover during time1~time2?
        3. finish_dict (pre jobs) --> Did pre jobs finished to release gpu resource?
    :param time1: start time.
    :param Err:
    :param finish_dict:
    :return: A list content increase time point if there is increasing.
            None if no gpu increase.
    """
    time = list()
    for gpu_id in Err.keys():
        t = (np.array(Err[gpu_id][0]) + np.array(Err[gpu_id][1])).tolist()
        time += t
    for job_id in finish_dict.keys():
        info = finish_dict[job_id]
        for k in range(len(info)):
            t = info[k][1]
            time.append(t)
    time.sort()
    for i in range(len(time)):
        if time1 < time[i]:
            return time[i]+1


def gen_metric():
    pass


def write_finish_dict(job_id, sub_time, end_time, gpus, hint, finish_dict):
    """
    To write submit time and real end time into finish_dict.
    Note that do not recover the old info, use [append] to write.
    :param finish_dict:
    :param gpus:
    :param end_time:
    :param job_id: str.
    :param sub_time: int.
    :param hint: str.
    :return:
    """
    if job_id not in finish_dict.keys():
        finish_dict[job_id] = list()
    finish_dict[job_id].append([sub_time, end_time, gpus, hint])


def run_jobs(Workloads, Err, scheduler, gpu_num, p_flag=False):
    # Input workloads:
    workloads = copy.deepcopy(Workloads)
    # Final result:
    finish_dict = {}
    # Waiting queue
    wait_q = {}

    next_job = None

    # while not (len(workloads) == 0 and len(wait_q) == 0): # while 0th
    while not (len(workloads) == 0 and len(wait_q) == 0):  # while 1st
        # print("workloads len: %d, wait_q len: %d." % (len(workloads), len(wait_q)))
        scheduler.order(wait_q)  # 更新所有作业的score， 包括wait_q & workloads
        # print("debug: wait_q=",wait_q)
        # print("debug: finish=", finish_dict)
        if next_job is not None: # br1-p
            job_id = next_job
            sub_time = workloads[job_id]["submit_time"]
            wait_or_workload = "workload"
            if p_flag:
                print("[Info]: Job %s in workflow will be submit because of its high priority." % job_id)
        elif len(wait_q) == 0:  # br1-no
            job_id = get_first_job_in_workloads(workloads)
            sub_time = workloads[job_id]["submit_time"]
            wait_or_workload = "workload"
            if p_flag:
                print("[Info]: Waiting queue is empty. Job %s in workflow will be submit." % job_id)
        else:  # br1-y
            last_sub_time = get_last_sub_time(finish_dict)
            if len(workloads) == 0:
                first_job_time_in_workloads = float('inf')
            else:
                first_job_time_in_workloads = get_first_job_time_in_workloads(workloads)
            gpu_increase_time_list = gpu_increase_in_period(last_sub_time, first_job_time_in_workloads, Err,
                                                            finish_dict)
            if gpu_increase_time_list is None:  # br2-no
                job_id = get_first_job_in_workloads(workloads)
                sub_time = workloads[job_id]["submit_time"]
                wait_or_workload = "workload"
            else:  # br2-y
                able_job_in_wait_q, gpu_release_time = \
                    get_able_job_in_wait_q(time_points=gpu_increase_time_list, wait_q=wait_q, finish_dict=finish_dict,
                                           Err=Err, gpu_num=gpu_num)
                if able_job_in_wait_q is None:  # br3-no
                    job_id = get_first_job_in_workloads(workloads)
                    sub_time = workloads[job_id]["submit_time"]
                    wait_or_workload = "workload"
                else:  # br3-y
                    job_id = able_job_in_wait_q
                    wait_q[job_id]["submit_time"] = gpu_release_time
                    sub_time = wait_q[job_id]["submit_time"]
                    wait_or_workload = "wait"
        while True:  # while 2nd
            free_gpu_list = get_free_gpu(sub_time, finish_dict, Err, gpu_num)
            # print("free gpus:", free_gpu_list, "  sub time:", sub_time)
            job_info = workloads[job_id] if wait_or_workload == "workload" else wait_q[job_id]
            if len(free_gpu_list) < job_info["GPU_num"]:  # br4-no
                if scheduler.backfill():  # br7-back
                    wait_q[job_id] = workloads[job_id]
                    del workloads[job_id]
                    if p_flag:
                        print(
                            "[Info]: BACKFILL - Moved job %s from workloads to wait_q because there is not enough GPU resource." % job_id)
                    next_job = None
                    break  # break while 2nd
                else: # br7-no
                    job_info["submit_time"] = next_gpu_increase_time(sub_time, Err, finish_dict)
                    if p_flag:
                        print(
                            "[Info]: NO-BACKFILL - Moved job %s from workloads to wait_q because there is not enough GPU resource." % job_id)
                    break
            else:  # br4-y
                # job_info = workloads[job_id] if wait_or_workload == "workload" else wait_q[job_id]
                gpu_ids = scheduler.place(free_gpu_list, job_info)
                err_or_preempt_flag, interrupt_time, preempt_id = get_err_or_preempt(job_info, gpu_ids, Err, workloads,
                                                                                     scheduler.preempt())
                if err_or_preempt_flag is None:  # br5-no
                    if p_flag:
                        print("[Info]: Job %s has no interruption during running period." % job_id)
                    end_time = wait_q[job_id]["submit_time"] + wait_q[job_id][
                        "running_time"] if wait_or_workload == "wait" \
                        else (workloads[job_id]["submit_time"] + workloads[job_id]["running_time"])
                    write_finish_dict(job_id, sub_time, end_time, gpu_ids, "None", finish_dict)
                    if wait_or_workload == "wait":
                        del wait_q[job_id]
                        if p_flag:
                            print("[Info]: Moved job %s from wait_q to finish_dict." % job_id)
                    elif wait_or_workload == "workload":
                        del workloads[job_id]
                        if p_flag:
                            print("[Info]: Moved job %s from workloads to finish_dict." % job_id)
                    next_job = None
                    break  # break while 2nd
                elif err_or_preempt_flag == "preempt":  # br5-p
                    if p_flag:
                        print("[Info]: Job %s is preempted by other jobs when time=%d." % (job_id, interrupt_time))
                    end_time = interrupt_time
                    write_finish_dict(job_id=job_id, sub_time=sub_time, end_time=end_time, gpus=gpu_ids, hint="preempt",
                                      finish_dict=finish_dict)
                    if wait_or_workload == "wait":
                        wait_q[job_id]["running_time"] -= end_time - sub_time
                        wait_q[job_id]["preempt_times"] += 1
                    elif wait_or_workload == "workload":
                        workloads[job_id]["running_time"] -= end_time - sub_time
                        workloads[job_id]["preempt_times"] += 1
                        wait_q[job_id] = workloads[job_id]
                        del workloads[job_id]
                        if p_flag:
                            print("[Info]: Moved job %s from workload to wait_q because of preemption." % job_id)
                    else:
                        raise ValueError("Job: %s's wait_or_workload != 'wait' or 'workload'." % job_id)
                    next_job = preempt_id
                    if p_flag:
                        print("[Info]: The preemption job is %s." % next_job)
                    break  # break while 2nd
                else:  # br5-e
                    if p_flag:
                        print("[Info]: Job %s is suspended because of device error when time=%d." % (job_id, interrupt_time))
                    end_time = interrupt_time
                    write_finish_dict(job_id=job_id, sub_time=sub_time, end_time=end_time, gpus=gpu_ids, hint="err",
                                      finish_dict=finish_dict)
                    if wait_or_workload == "wait":
                        wait_q[job_id]["running_time"] -= end_time - sub_time
                        wait_q[job_id]["err_times"] += 1
                    elif wait_or_workload == "workload":
                        workloads[job_id]["running_time"] -= end_time - sub_time
                        workloads[job_id]["err_times"] += 1
                        wait_q[job_id] = workloads[job_id]
                        del workloads[job_id]
                        if p_flag:
                            print("[Info]: Moved job %s from workload to wait_q because of device error." % job_id)
                    else:
                        raise ValueError("Job: %s's wait_or_workload != 'wait' or 'workload'." % job_id)
                    restart_now = scheduler.restart(job_id)
                    if restart_now is False:  # br6-no
                        next_job = None
                        break  # break while 2nd
                    else:  # br6-y
                        wait_q[job_id]["submit_time"] = end_time
                        next_job = None
                        continue  # back to while 2nd
    return finish_dict
