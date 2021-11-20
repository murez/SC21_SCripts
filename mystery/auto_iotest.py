import os

def iotest(rank, mb):
    run = "mpirun -n {RANKS} python bin/exec.py run.id=task1 mode=iotest run.iterations=100 run.minibatch_size={MB} data.data_directory=/data/datasets/mystery/ data.file=cosmic_tagging_train.h5 run.compute_mode=CPU > ans.txt".format(RANKS=rank, MB=mb)
    os.system(run)
    with open("ans.txt", "r") as f:
        iotime = 1000
        throughput = 0
        lines = f.readlines()
        for line in lines:
            if "Total IO Time:" in line:
                iotime = float(line.split(':')[-1].split('s')[0])
            if "Throughput:" in line:
                throughput = float(line.split(':')[-1])
        return iotime, throughput

iotest_result = []

for logrank in range(5, 10):
    rank = int(2**logrank)
    for alpha in range(1, 4):
        mb = rank * alpha
        iotime, throughput = iotest(rank, mb)
        print("rank: {} minibatch: {} iotime: {} throughput: {}".format(rank, mb, iotime, throughput))
        iotest_result.append([rank, mb, iotime, throughput])

with open("iotest_result.txt", "w") as f:
    for i in iotest_result:
        f.write("rank: {} minibatch: {} iotime: {} throughput: {}\n".format(i[0], i[1], i[2], i[3]))
