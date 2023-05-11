class replace_attr:
    def __init__(self, obj, **kwargs) -> None:
        self.obj = obj
        assert len(kwargs) == 1
        for name, value in kwargs.items():
            self.attr_name, self.new_attr = name, value
        self.old_attr = getattr(self.obj, self.attr_name)

    def __enter__(self):
        setattr(self.obj, self.attr_name, self.new_attr)
        return self

    def __exit__(self, *args, **kwargs):
        setattr(self.obj, self.attr_name, self.old_attr)


def merge_dicts(list_of_dict):
    import pandas as pd
    return pd.DataFrame(list(list_of_dict)).to_dict(orient="list")
