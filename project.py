import os
from time import sleep

import fire


def RUN(x):
    os.system(x)


class Project:
    def run(self, competition):
        RUN(f"python -m {competition}")

    def submit(self, competition):
        RUN(
            f'kaggle competitions submit -c {competition} -f working/submission.csv -m "{competition}"'
        )
        sleep(5)
        RUN(f"kaggle competitions submissions {competition}")


if __name__ == "__main__":
    fire.Fire(Project)
