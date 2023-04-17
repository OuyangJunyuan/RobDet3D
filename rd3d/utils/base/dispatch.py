class Field:
    def __init__(self, attrs=None):
        self.attrs = attrs or []

    def __getattr__(self, item):
        return Field(self.attrs + [item])

    def __iter__(self):
        yield from self.attrs


def dispatch(func):
    import warnings
    from functools import wraps
    from collections import defaultdict
    assert func.__annotations__
    dispatch.cache = getattr(dispatch, 'cache', defaultdict(dict))

    for k, v in func.__annotations__.items():
        if isinstance(v, Field):
            func.__dispatch_info__ = func.__code__.co_varnames.index(k), list(v)
            dispatch.cache[func.__name__][func.__dispatch_info__[1].pop(-1)] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        state = args[func.__dispatch_info__[0]]
        for name in func.__dispatch_info__[1]: state = getattr(state, name)
        impl = dispatch.cache[func.__name__].get(str(state).lower(),
                                                 lambda *args, **kwargs: warnings.warn(f'dispatch->{state} no found'))
        return impl(*args, **kwargs)

    return wrapper


when = Field()


def test():
    from easydict import EasyDict as obj

    @dispatch
    def foo(arg1: when.state.train):
        print(foo.__name__, 'train')

    @dispatch
    def foo(arg1: when.state.test):
        print(foo.__name__, 'test')

    @dispatch
    def foo(arg1: when.state.true):
        print(foo.__name__, 'true')

    @dispatch
    def foo(arg1: when.state.false):
        print(foo.__name__, 'false')

    @dispatch
    def bar(arg1, arg2: when.cfg.remove.enable.true):
        print(bar.__name__, 'true')

    # @dispatch
    # def bar(arg1, arg2: when.cfg.remove.enable.false):
    #     print(bar.__name__, 'false')

    foo(obj(state='val'))
    foo(obj(state='test'))
    foo(obj(state='train'))
    foo(obj(state=True))
    foo(obj(state=False))
    bar(1, obj(dict(cfg=dict(remove=dict(enable=False)))))
    bar(2, obj(dict(cfg=dict(remove=dict(enable=True)))))


if __name__ == '__main__':
    test()
