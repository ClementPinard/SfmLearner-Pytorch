from blessings import Terminal
import progressbar
import sys


class TermLogger(object):
    def __init__(self, n_epochs, train_size, valid_size):
        self.n_epochs = n_epochs
        self.train_size = train_size
        self.valid_size = valid_size
        self.t = Terminal()
        s = 10
        e = 1   # epoch bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        h = self.t.height

        for i in range(10):
            print('')
        self.epoch_bar = progressbar.ProgressBar(max_value=n_epochs, fd=Writer(self.t, (0, h-s+e)))

        self.train_writer = Writer(self.t, (0, h-s+tr))
        self.train_bar_writer = Writer(self.t, (0, h-s+tr+1))

        self.valid_writer = Writer(self.t, (0, h-s+ts))
        self.valid_bar_writer = Writer(self.t, (0, h-s+ts+1))

        self.reset_train_bar()
        self.reset_valid_bar()

    def reset_train_bar(self):
        self.train_bar = progressbar.ProgressBar(max_value=self.train_size, fd=self.train_bar_writer)

    def reset_valid_bar(self):
        self.valid_bar = progressbar.ProgressBar(max_value=self.valid_size, fd=self.valid_bar_writer)


class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        with self.t.location(*self.location):
            sys.stdout.write("\033[K")
            print(string)

    def flush(self):
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, i=1, precision=3):
        self.meters = i
        self.precision = precision
        self.reset(self.meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.count = 0

    def update(self, val, n=1):
        if not isinstance(val, list):
            val = [val]
        assert(len(val) == self.meters)
        self.count += n
        for i,v in enumerate(val):
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ' '.join(['{:.{}f}'.format(v, self.precision) for v in self.val])
        avg = ' '.join(['{:.{}f}'.format(a, self.precision) for a in self.avg])
        return '{} ({})'.format(val, avg)
