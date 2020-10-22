class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def status(string):
    print(bcolors.OKCYAN+string+bcolors.ENDC)

def warning(string):
    print(bcolors.WARNING+string+bcolors.ENDC)

def error(string):
    print(bcolors+FAIL+string+bcolors.ENDC)
