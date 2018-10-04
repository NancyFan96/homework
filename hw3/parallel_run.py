import os
import time
from multiprocessing import Process
from run_dqn_lander import main as run_lander
from run_dqn_ram import main as run_ram
from run_dqn_atari import main as run_atari


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='lander')
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    args = parser.parse_args()

    target_func = None
    if args.exp_name == 'lander':
        target_func = run_lander
    elif args.exp_name == 'ram':
        target_func = run_ram
    elif args.exp_name == 'atari':
        target_func = run_atari

    processes = []

    for e in range(args.n_experiments):
        print('Running experiment [%d/%d]' % (e, args.n_experiments))

        # # Awkward hacky process runs, because Tensorflow does not like
        # # repeatedly calling train_AC in the same thread.
        p = Process(target=target_func, args=tuple())
        p.start()
        processes.append(p)
        # if you comment in the line below, then the loop will block
        # until this process finishes
        # p.join()

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
