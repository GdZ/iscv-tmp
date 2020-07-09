import time
# -----------------------------------------------------------------------------------
# global variable area
# DEBUG = True
DEBUG = False


# function
def is_debug():
    return DEBUG
# end


def logW(msg):
    if is_debug():
        print('{:.06f} [warning] {}'.format(time.time(), msg))


def logD(msg):
    if is_debug():
        print('{:.06f} [debug] {}'.format(time.time(), msg))


def logV(msg):
    print('{:.06f} [verbose] {}'.format(time.time(), msg))

