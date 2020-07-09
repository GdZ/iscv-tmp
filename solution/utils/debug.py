import time
# -----------------------------------------------------------------------------------
# global variable area
# DEBUG = True
DEBUG = False


# function
def isDebug():
    return DEBUG
# end


def logW(msg):
    if isDebug():
        print('{:.06f} [warning] {}'.format(time.time(), msg))


def logD(msg):
    if isDebug():
        print('{:.06f} [debug] {}'.format(time.time(), msg))


def logV(msg):
    print('{:.06f} [verbose] {}'.format(time.time(), msg))

