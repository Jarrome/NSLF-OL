
import tty,sys,select,termios

'''
https://blog.csdn.net/Callme_TeacherPi/article/details/124228502
'''

def getKey(settings):
    tty.setraw(sys.stdin.fileno())
    rlist = select.select([sys.stdin],[],[],0.1)

    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ""

    termios.tcsetattr(sys.stdin,termios.TCSADRAIN,settings)
    return key

if __name__ == '__main__':

    while(1):
        setting = termios.tcgetattr(sys.stdin)
        InPut = getKey(setting)
        if Input == 'w':
            print(InPut)
        else:
            break
