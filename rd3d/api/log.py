from accelerate import logging


def info_for_iterable_obj(func):
    from typing import Iterable, Sequence, ByteString, Mapping
    iterable = lambda x: isinstance(x, Iterable) and not isinstance(x, (str, ByteString))

    def make_table(data, title=None, deep=0, width=50):
        import prettytable
        import textwrap
        if iterable(data):
            kvs = list(data.items() if isinstance(data, Mapping) else enumerate(data))
            """有标题 | 是字典 | 元素全可迭代 ==> 以表格形式输出"""
            if title or isinstance(data, Mapping) or sum([isinstance(v, Iterable) for _, v in kvs]):
                tb = prettytable.PrettyTable(title, header=title is not None)
                tb.set_style(prettytable.SINGLE_BORDER)
                tb.add_rows([[k, make_table(data=v, deep=deep + 1, width=width)] for k, v in kvs])
                return tb.get_string(fields=tb.field_names[1:], border=False) \
                    if isinstance(data, Sequence) and deep else tb.get_string()
        return textwrap.fill(data.__str__(), width=width) if width else data.__str__()

    def wrapper(msg, *args, **kwargs):
        if iterable(msg):
            f = kwargs.pop('exclude', None)
            width = kwargs.pop('width', 0)
            msg = msg if f is None else {k: v for k, v in msg.items() if k not in f}
            for msg in make_table(msg, title=kwargs.pop('title', None), width=width).split(sep='\n'):
                func(msg, *args, **kwargs)
        else:
            func(msg, *args, **kwargs)

    return wrapper


def create_logger(log_file=None, stderr=True):
    if not hasattr(create_logger, 'logger'):
        logger = logging.get_logger(__name__)

        logger.logger.setLevel(logging.logging.INFO)
        formatter = logging.logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')

        """log to stderr"""
        if stderr:
            console = logging.logging.StreamHandler()
            console.setLevel(logging.logging.INFO)
            console.setFormatter(formatter)
            logger.logger.addHandler(console)

        """"log to file"""
        if log_file is not None:
            file_handler = logging.logging.FileHandler(filename=log_file)
            file_handler.setLevel(logging.logging.INFO)
            file_handler.setFormatter(formatter)
            logger.logger.addHandler(file_handler)

        logger.logger.propagate = False  # 否则会让root logger也一起发出日志
        logger.info = info_for_iterable_obj(logger.info)
        create_logger.logger = logger
    return create_logger.logger


class log_to_file:
    def __init__(self) -> None:
        pass

    def __enter__(self) -> None:
        for handle in create_logger.logger.logger.handlers:
            if not isinstance(handle, logging.logging.FileHandler):
                handle.setLevel(logging.logging.FATAL)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        for handle in create_logger.logger.logger.handlers:
            if not isinstance(handle, logging.logging.FileHandler):
                handle.setLevel(logging.logging.INFO)
