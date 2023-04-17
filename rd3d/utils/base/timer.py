import time

try:
    import torch

    synchronize = torch.cuda.synchronize()
except ImportError:
    synchronize = lambda: None

TIMER_ENABLE = True


class ScopeTimer:
    """
    Examples:

        for i in range(2):
            with TimeMeasurement("demo: ") as t:
                1+1
                t.tail = " 1+1"
            print(t.duration)

        ---
        demo: 0.035ms(1) 1+1
        0.03528594970703125
        demo: 0.008ms(2) 1+1
        0.008106231689453125

    """

    context = {}

    def __init__(self, title, average=False, verbose=True, enable=None):
        self.title = title
        self.verbose = verbose
        self.average = average
        self.enable = TIMER_ENABLE if enable is None else enable

        self.duration = None
        self.tail = None
        if self.average and self.title not in self.context:
            self.context[self.title] = [0, 0]

    def __del__(self):
        pass

    def __enter__(self):
        if self.enable:
            torch.cuda.synchronize()
            self.t1 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            torch.cuda.synchronize()
            self.duration = (time.time() - self.t1) * 1e3
            if self.average:
                context = self.context[self.title]
                context[0] += self.duration  # accumulated duration
                context[1] += 1  # recording times
                if context[1] == 2:  # 除去第一次含有jit编译时间的测量
                    context[0] = 2 * self.duration
                self.duration = context[0] / context[1]
            if self.verbose:
                content = '{}{:.3f}ms'.format(self.title, self.duration)
                content += f"({self.context[self.title][1]})" if self.average else ''
                content += (self.tail if self.tail is not None else '')
                print(content, flush=True)
        return exc_type is None


def measure_time(title=None, *args1, **kwargs1):
    """

    Examples:

        @measure_time()
        def hellow(str1):
            print(str1)

        hellow("123")

        ---
        123
        hellow: 0.066ms(1)

    """

    def decorator(func):
        def handler(*args2, **kwargs2):
            with ScopeTimer(title=f"{func.__name__}: " if title is None else title, *args1, *kwargs1) as t:
                ret = func(*args2, **kwargs2)
            return ret

        return handler

    return decorator


def test():
    with ScopeTimer('loop1: '):
        for i in range(1000000):
            a = 1 + 1
    with ScopeTimer('loop2: ', verbose=False) as t:
        for i in range(1000000):
            a = 1 + 1

    print(t.duration)

    @measure_time()
    def do_something():
        for _ in range(10):
            with ScopeTimer('loop3: ', average=True) as t:
                for i in range(1000000):
                    a = 1 + 1

    do_something()

if __name__ == '__main__':
    test()
