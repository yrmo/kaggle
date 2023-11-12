import os
import shlex
import subprocess
from time import sleep
from typing import Final

import fire  # type: ignore

RUN = lambda x: os.system(x)


def RUN_OUT(x):
    result = subprocess.run(
        x, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print(f"{result.stdout}")
    print(f"{result.stderr}")


class Project:
    def run(self, competition):
        RUN(f"export SUBMIT=0; python -m {competition}")

    def submit(self, competition):
        RUN_OUT(f"export SUBMIT=1; python -m {competition}")
        RUN(
            f'kaggle competitions submit -c {competition} -f working/submission.csv -m "{competition}"'
        )
        sleep(5)
        RUN(f"kaggle competitions submissions {competition} | head -n 5")

    def type(self):
        RUN("python -m mypy --install-types")
        RUN("python -m mypy .")

    def data(self, competition):
        COMP_DIR: Final = f"input/{competition}"
        os.makedirs(COMP_DIR, exist_ok=True)

        RUN(f"kaggle competitions download -c {competition} -p {COMP_DIR}")

        RUN(f"unzip -o {COMP_DIR}/*.zip -d {COMP_DIR}")


if __name__ == "__main__":
    fire.Fire(Project)
