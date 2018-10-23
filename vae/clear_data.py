import os
from load_hparams import hparams

print('WARNING: this script deletes data')

pathes = [
    'checkpoints_path',
    'export_path',
    'pairs_path',
    'answers_path',
    ]

class _Getch:
    """Gets a single character from standard input.  Does not echo to the screen."""
    def __init__(self):
        try:
            self.impl = _GetchWindows()
        except ImportError:
            self.impl = _GetchUnix()

    def __call__(self): return self.impl()
class _GetchUnix:
    def __init__(self):
        import tty, sys
    def __call__(self):
        import sys, tty, termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
class _GetchWindows:
    def __init__(self):
        import msvcrt

    def __call__(self):
        import msvcrt
        return msvcrt.getch()
getch = _Getch()

for p in pathes:
    if p in hparams:
        print('\n%s: %s' % (p, hparams[p]))
        #inputs = input('clear data in %s? (y/n) ' % p)
        print('clear data in %s? (y/n/q)' % p)
        inputs = getch()
        if inputs == 'y':
            os.system('rm -r %s' % hparams[p])
        if inputs in ['q', '']:
            exit()





