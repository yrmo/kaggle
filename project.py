import os
from time import sleep

import fire  # type: ignore

RUN = lambda x: os.system(x)


class Project:
    def run(self, competition):
        RUN(f"export SUBMIT=0; python -m {competition}")

    def submit(self, competition):
        RUN(f"export SUBMIT=1; python -m {competition}")
        RUN(
            f'kaggle competitions submit -c {competition} -f working/submission.csv -m "{competition}"'
        )
        sleep(5)
        RUN(f"kaggle competitions submissions {competition}")

    def type(self):
        RUN("python -m mypy --install-types")
        RUN("python -m mypy .")


if __name__ == "__main__":
    fire.Fire(Project)
