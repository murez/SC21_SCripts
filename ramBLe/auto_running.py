import subprocess
import os
import time

import pandas
import glob
import tqdm
import logging
import heapq
import json
from re import search
import uuid
import multiprocessing as mp
import numpy as np
import time

from typing import List


class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


hostfile_dir = '/nfs/cluster/hostfiles/'
output_dir = '/nfs/cluster/result'
ramble_binary = "/nfs/cluster/ramBLe_hpcx/ramble"

### for debugging only
# hostfile_dir = "F:\\"

FORMAT = '[%(status)-7s] %(method)-10s %(asctime)s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger('task')

resource_lock = mp.Lock()

task_queue = PriorityQueue()


def can_allocate(n: int, cpu_resource_remain) -> bool:
    """
    :param n: cpu cores
    :return: whether remain cpu resources can allocate e
    """
    # print(cpu_resource_remain)
    resource_lock.acquire()
    res = False
    if n <= 32:
        for node in cpu_resource_remain.keys():
            if len(cpu_resource_remain[node]) >= n:
                res = True
                break
    elif n > 64:
        res = sum([len(x) for x in cpu_resource_remain.values()]) >= n
    else:
        x = None
        for node in cpu_resource_remain.keys():
            if len(cpu_resource_remain[node]) >= 32:
                x = node
                break
        if x is not None:
            for node in cpu_resource_remain.keys():
                if node != x:
                    if len(cpu_resource_remain[node]) >= n - 32:
                        res = True
                        break

    resource_lock.release()
    return res


def generate_rankfile(n: int, filename: str, cpu_resource_remain) -> dict:
    """
        :param n: core number to request
        :param filename: hostfile path
        :return: rankfile dict
    """
    resource_lock.acquire()
    rankfile_dict = {}
    if n <= 32:
        for node in cpu_resource_remain.keys():
            if len(cpu_resource_remain[node]) >= n:
                rankfile_dict[node] = set(sorted(list(cpu_resource_remain[node]))[:n])
                cpu_resource_remain[node] -= rankfile_dict[node]
                break
    elif n > 64:
        n_copied = n
        for node in cpu_resource_remain.keys():
            now_core_num = len(cpu_resource_remain[node])
            if now_core_num == 0:
                continue
            if n_copied >= now_core_num:
                rankfile_dict[node] = set(sorted(list(cpu_resource_remain[node])))
                cpu_resource_remain[node] -= rankfile_dict[node]
                n_copied -= now_core_num
            else:
                rankfile_dict[node] = set(sorted(list(cpu_resource_remain[node]))[:n_copied])
                cpu_resource_remain[node] -= rankfile_dict[node]
                n_copied = 0
                break
    else:
        x = None
        n_copied = n
        for node in cpu_resource_remain.keys():
            if len(cpu_resource_remain[node]) == 32:
                x = node
                rankfile_dict[node] = set(sorted(list(cpu_resource_remain[node])))
                cpu_resource_remain[node] -= rankfile_dict[node]
                n_copied -= 32
                break
        for node in cpu_resource_remain.keys():
            if node != x:
                if len(cpu_resource_remain[node]) >= n_copied:
                    rankfile_dict[node] = set(sorted(list(cpu_resource_remain[node]))[:n_copied])
                    cpu_resource_remain[node] -= rankfile_dict[node]
                    n_copied = 0
                    break
    with open(filename, 'w') as f:
        rank_count = 0
        for node in rankfile_dict.keys():
            for core in rankfile_dict[node]:
                f.write("rank {rank_num}={hostname} slots={core_num}\n".format(rank_num=rank_count, hostname=node,
                                                                               core_num=core))
                rank_count += 1
    resource_lock.release()
    return rankfile_dict


def generate_hostfile(n: int, filename: str, cpu_resource_remain) -> dict:
    """
    :param n: core number to request
    :param filename: hostfile path
    :return: hostfile dict
    """
    resource_lock.acquire()
    hostfile_dict = {}
    for node in cpu_resource_remain.keys():
        now_core_num = cpu_resource_remain[node]
        if now_core_num == 0:
            continue
        if n > now_core_num:
            hostfile_dict[node] = now_core_num
            cpu_resource_remain[node] -= now_core_num
            n -= now_core_num
        else:
            hostfile_dict[node] = n
            cpu_resource_remain[node] -= n
            n = 0
            break
    with open(filename, 'w') as f:
        for node in hostfile_dict.keys():
            f.write("{} slots={}\n".format(node, hostfile_dict[node]))
    # print(cpu_resource_remain)
    resource_lock.release()
    return hostfile_dict


def finish_compute(slot_dict: dict, cpu_resource_remain):
    """
    :param slot_dict: hostfile dict
    :return:
    """
    resource_lock.acquire()
    for node in slot_dict.keys():
        cpu_resource_remain[node] = cpu_resource_remain[node] | slot_dict[node]
    # print(cpu_resource_remain)
    resource_lock.release()


task_check_set = {"name", "input", "cores", "algorithm", "level", "flags", "sep", "n", "m", "repeat"}


def read_task_from_file(filename):
    """
    :param filename: json task file name
        input: "/nfs/scratch/C1_discretized.tsv" filepath
        output: "/nfs/scratch/" result save dir
        cores: 128 int
        level: 0 int
        flags: "-d" string like "-c -v -i -d"
        sep: "$'\t'" means tab
        n: 0 int
        m: 0 int
        repeat: 1 int
    :return: task list
    """
    global ramble_binary
    global output_dir
    with open(filename, 'r') as f:
        file_json = json.load(f)
        ramble_binary = file_json['ramble_binary']
        output_dir = file_json['result_directory']
        task_list = file_json['tasks']
        # check task in task list have all
        for index, task in enumerate(task_list):
            if set(task.keys()) == task_check_set:
                d = {"status": 'success', 'method': 'read file'}
                logger.info("read task No.{} successfully".format(index), extra=d)
            else:
                d = {"status": 'failed', 'method': 'read file'}
                logger.error("read task No.{} failed please check and try again".format(index), extra=d)
                raise ValueError
    return task_list


def generate_command(task: dict, cpu_resource_remain):
    """
    allocate cpu resources and generate commands
    :param task: single task dict from the json file
    :return:
    """
    # create hostfile
    task_name = task['name']
    task_algo = task['algorithm']
    task_core = task['cores']
    hostfilename = "{name}_{core}_{algo}.hostfile".format(
        name=task_name,
        core=task_core,
        algo=task_algo
    )
    hostfile_path = os.path.join(hostfile_dir, hostfilename)
    # slot_dict = generate_hostfile(task_core, hostfile_path, cpu_resource_remain)

    slot_dict = generate_rankfile(task_core, hostfile_path, cpu_resource_remain)

    # generate command list
    command_list = []
    for repeat_times in range(task['repeat']):
        output_filename = "{name}_{core}_{algo}_{repeat}.dot".format(
            name=task_name,
            core=task_core,
            algo=task_algo,
            repeat=repeat_times
        )
        output_filepath = os.path.join(output_dir, output_filename)
        timer_filename = "{name}_{core}_{algo}_{repeat}.timer".format(
            name=task_name,
            core=task_core,
            algo=task_algo,
            repeat=repeat_times
        )
        timer_filepath = os.path.join(output_dir, timer_filename)
        command = """mpirun -np {cores} \
        -rf  {hostfile} \
        -x MXM_RDMA_PORTS=mlx5_0:1 \
        -mca btl_openib_if_include mlx5_0:1 \
        -x UCX_NET_DEVICES=mlx5_0:1 \
        {ramble} -f {input} -n {n} -m {m} {flag} -s {sep} -o {output} -a {algo} -r --warmup --hostnames""".format(
            cores=task_core,
            hostfile=hostfile_path,
            ramble=ramble_binary,
            input=task['input'],
            n=task['n'],
            m=task['m'],
            flag=task['flags'],
            sep=task['sep'],
            output=output_filepath,
            algo=task_algo
        )
        command_list.append((command, timer_filepath))
    return slot_dict, command_list


def get_runtime(action, output, required=True):
    float_pattern = r'((?:(?:\d*\.\d+)|(?:\d+\.?))(?:[Ee][+-]?\d+)?)'
    pattern = 'Time taken in %s: %s' % (action, float_pattern)
    match = search(pattern, output)
    if required:
        return float(match.group(1))
    else:
        return float(match.group(1) if match is not None else 0)


def parse_runtimes(process, result_path):
    output = ''
    for line in iter(process.stdout.readline, b''):
        line = line.decode('utf-8')
        output += line
    # write into result file
    with open(result_path, 'w') as f:
        # print(output)
        f.write(output)
    # optional runtimes
    warmup = get_runtime('warming up MPI', output, required=False)
    redistributing = get_runtime('redistributing', output, required=False)
    blankets = get_runtime('getting the blankets', output, required=False)
    symmetry = get_runtime('symmetry correcting the blankets', output, required=False)
    sync = get_runtime('synchronizing the blankets', output, required=False)
    neighbors = get_runtime('getting the neighbors', output, required=False)
    direction = get_runtime('directing the edges', output, required=False)
    gsquare = get_runtime('G-square computations', output, required=False)
    mxx = get_runtime('mxx calls', output, required=False)
    # required runtimes
    reading = get_runtime('reading the file', output, required=True)
    network = get_runtime('getting the network', output, required=True)
    writing = get_runtime('writing the network', output, required=True)
    return [warmup, reading, redistributing, blankets, symmetry, sync, neighbors, direction, mxx, gsquare, network,
            writing]


def run_task(task: dict, cpu_resource_remain):
    slot_dict, command_list = generate_command(task, cpu_resource_remain)
    d = {"status": 'success', 'method': 'allocate'}
    logger.info("allocate {} cores successfully".format(task["cores"]), extra=d)
    result_list = []
    for command, result_path in command_list:
        d = {"status": 'process', 'method': 'running'}
        logger.info("start {} cores of {}".format(task["cores"], task["name"]), extra=d)
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        timer_list = parse_runtimes(process, result_path)
        d = {"status": 'success', 'method': 'running'}
        logger.info("success {} cores of {}".format(task["cores"], task["name"]), extra=d)
        result_list.append(timer_list)
    finish_compute(slot_dict, cpu_resource_remain)
    result_list_np = np.array(result_list)
    timer_filename = "{name}_{core}_{algo}_average.timer".format(
        name=task["name"],
        core=task["cores"],
        algo=task["algorithm"]
    )
    timer_filepath = os.path.join(output_dir, timer_filename)
    average_result = np.average(result_list_np, axis=0)
    average_res = {
        "warmup": average_result[0],
        "reading": average_result[1],
        "redistributing": average_result[2],
        "blankets": average_result[3],
        "symmetry": average_result[4],
        "sync": average_result[5],
        "neighbors": average_result[6],
        "direction": average_result[7],
        "mxx": average_result[8],
        "gsquare": average_result[9],
        "network": average_result[10],
        "writing": average_result[11],
    }
    with open(timer_filepath, 'w') as f:
        json.dump(average_res, f)


if __name__ == '__main__':
    manager = mp.Manager()

    cpu_resource_remain = manager.dict()

    cpu_resource_remain["hpc-cluster-node-1"] = set(range(4, 36))
    cpu_resource_remain["hpc-cluster-node-2"] = set(range(4, 36))
    cpu_resource_remain["hpc-cluster-node-3"] = set(range(4, 36))
    cpu_resource_remain["hpc-cluster-node-4"] = set(range(4, 36))

    task_list = read_task_from_file("tasks.json")
    for task in task_list:
        task_queue.push(task, task['level'])
    now_task = task_queue.pop()
    while not task_queue.isEmpty():
        if can_allocate(now_task["cores"], cpu_resource_remain):
            d = {"status": 'success', 'method': 'allocate'}
            logger.info("{} cores can be allocated".format(now_task["cores"]), extra=d)
            mp.Process(target=run_task, args=(now_task, cpu_resource_remain)).start()
            now_task = task_queue.pop()

        time.sleep(60)
