import random
import numpy as np
import copy


class demo_scheduler:
    def __init__(self):
        pass
    def init_workflow(self, workloads):
        return workloads.keys()

    def order(self, wait_q, workload=None):
        # 作业得分策略，即排队策略，高分代表高优先级，得分必须是正整数。
        # 可更新所有作业的score，包括wait_q & workloads。也可以只更新wait_q中作业得分。
        # e.g. 每次都给wait_q中的所有作业得分*1.5倍：
        if len(wait_q) != 0:
            for job_id in wait_q.keys():
                wait_q[job_id]["score"] *= 1.5

    def place(sef, free_gpu_list, job_info):
        # 放置策略
        # e.g. 随机放置策略：
        num = job_info["GPU_num"]
        gpu_ids = random.sample(free_gpu_list, num)
        return gpu_ids

    def restart(self,job_id):
        # 出错重启策略
        # e.g. 不重启:
        return False

    def preempt(self):
        # 是否抢占（动态调度）, 抢占返回True，不抢占返回False
        return True

    def backfill(self):
        return False
