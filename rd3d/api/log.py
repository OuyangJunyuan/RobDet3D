from accelerate import logging as warp

logging = warp.logging


class LogIterableObject:
    def __init__(self, f):
        self.f = f

    @property
    def iterable(self):
        from typing import Iterable, ByteString
        return lambda x: isinstance(x, Iterable) and not isinstance(x, (str, ByteString))

    def make_table(self, data, title=None, deep=0, width=50):
        import textwrap
        import prettytable
        from typing import Iterable, Sequence, Mapping
        if self.iterable(data):
            kvs = list(data.items() if isinstance(data, Mapping) else enumerate(data))
            """有标题 | 是字典 | 元素全可迭代 ==> 以表格形式输出"""
            if title or isinstance(data, Mapping) or sum([isinstance(v, Iterable) for _, v in kvs]):
                tb = prettytable.PrettyTable(title, header=title is not None)
                tb.set_style(prettytable.SINGLE_BORDER)
                tb.add_rows([[k, self.make_table(data=v, deep=deep + 1, width=width)] for k, v in kvs])
                return tb.get_string(fields=tb.field_names[1:], border=False) \
                    if isinstance(data, Sequence) and deep else tb.get_string()
        return textwrap.fill(data.__str__(), width=width) if width else data.__str__()

    def log(self, level, msg, *args, **kwargs):
        if self.iterable(msg):
            f = kwargs.pop('exclude', None)
            width = kwargs.pop('width', 0)
            msg = msg if f is None else {k: v for k, v in msg.items() if k not in f}
            msgs = self.make_table(msg, title=kwargs.pop('title', None), width=width).split(sep='\n')
            for msg in msgs:
                self.f(level, msg, *args, **kwargs)
        else:
            self.f(level, msg, *args, **kwargs)

    def __call__(self, level, msg, *args, **kwargs):
        if self.iterable(msg):
            f = kwargs.pop('exclude', None)
            width = kwargs.pop('width', 0)
            msg = msg if f is None else {k: v for k, v in msg.items() if k not in f}
            msgs = self.make_table(msg, title=kwargs.pop('title', None), width=width).split(sep='\n')
            for msg in msgs:
                self.f(level, msg, *args, **kwargs)
        else:
            self.f(level, msg, *args, **kwargs)


def create_logger(name=None, log_file=None, stderr=True, level=logging.INFO):
    def get_formatter():
        name_header = '' if name is None else name
        return '[' + name_header[:3] + ' %(asctime)s %(levelname)s' + ']' + ' %(message)s'

    def stream_handler():
        import sys
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

    def file_handler():
        handler = logging.FileHandler(filename=log_file)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

    formatter = logging.Formatter(get_formatter())
    warped_logger = warp.get_logger(name)
    logger = warped_logger.logger

    logger.setLevel(level)
    logger.propagate = False  # disable the message propagation to the root logger
    warped_logger.log = LogIterableObject(warped_logger.log)

    if stderr:
        logger.addHandler(stream_handler())
    if log_file:
        logger.addHandler(file_handler())
    return warped_logger


root_log = create_logger(logging.WARNING)
