# coding: utf-8
# Copyright (c) Materials Center Leoben Forschung GmbH (MCL)

"""
Module containing checker methods
"""

results = {"burgers_vector": 2.5137028325558415, "a": 3.5549126375761357}


class bcolors:
    """
    Colors for the printing
    """

    HEADER = "\033[94m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def check(**kwargs):
    """
    Check result and show relative error
    """
    for k, v in kwargs.items():
        if k in results:
            rel_error = abs(results[k] - v) / abs(results[k])
            print(
                f"{bcolors.HEADER}Relative error of your solution is: {rel_error * 100} %{bcolors.ENDC}"
            )
        else:
            print(f"{bcolors.FAIL}{k} doesn't exists{bcolors.ENDC}")
