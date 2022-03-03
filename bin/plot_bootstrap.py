import os


def ks_exp():
    n_patch_sizes = [1, 5, 9, 13, 17, 21, 25, 29, 33, 65, 129]
    n_levels = [8, 6, 5, 4, 4, 3, 3, 3, 3, 2, 1]

    assert(len(n_patch_sizes) == len(n_levels))

    for idx, patch in enumerate(n_patch_sizes):
        num_levels = n_levels[idx]
        for level in range(num_levels):
            os.system(f'python bin/anonymity_score.py --level {level + 1} --patchsize {patch} --n_patients 100 --num_resample 10')



if __name__ == "__main__":
    ks_exp()







#os.system("python bin/anonymity_score.py --level 2 --patchsize 33 --n_patients 30 --num_resample 10")