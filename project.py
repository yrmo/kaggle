import os
import shlex
import subprocess
from functools import wraps
from time import sleep
from typing import Final

import fire  # type: ignore


def echo(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(*args, **kwargs)
        return func(*args, **kwargs)

    return wrapper


@echo
def RUN(x):
    os.system(x)


class Project:
    def check_abbreviations(self, competition):
        abbreviations = {
            "house-prices": "house-prices-advanced-regression-techniques",
        }
        if "_" in competition:
            competition = competition.replace("_", "-")
        if competition in abbreviations.keys():
            print(f"Abbreviation found: {competition} -> {abbreviations[competition]}")
            return abbreviations[competition]
        return competition

    def run(self, competition):
        RUN(f"export SUBMIT=0; python -m {competition}")

    def submit(self, competition):
        RUN(f"export SUBMIT=1; python -m {competition}")
        competition = self.check_abbreviations(competition)
        RUN(
            f'kaggle competitions submit -c {competition} -f working/submission.csv -m "{competition}"'
        )
        sleep(5)
        RUN(f"kaggle competitions submissions {competition} | head -n 5")

    def type(self):
        RUN("python -m mypy --install-types")
        RUN("python -m mypy .")

    def data(self, competition):
        competition = self.check_abbreviations(competition)
        COMP_DIR: Final = f"input/{competition}"
        os.makedirs(COMP_DIR, exist_ok=True)

        RUN(f"kaggle competitions download -c {competition} -p {COMP_DIR}")

        RUN(f"unzip -o {COMP_DIR}/*.zip -d {COMP_DIR}")


if __name__ == "__main__":
    fire.Fire(Project)
