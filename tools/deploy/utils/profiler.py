import numpy as np
import tensorrt as trt


class MyProfiler(trt.IProfiler):
    def __init__(self, collects):
        from collections import defaultdict
        trt.IProfiler.__init__(self)
        self.infos = defaultdict(list)
        self.name_of_first = None
        self.infer_times = 0
        self.collects = collects
        self.collect_infos = {k: [] for k in self.collects + ["Others"]}

    def report_layer_time(self, layer_name, ms):
        if self.name_of_first is None:
            self.name_of_first = layer_name
            self.infer_times = 1
        elif self.name_of_first == layer_name:
            self.infer_times += 1
        self.infos[layer_name].append(ms)

    def print(self):
        total = []
        for v in zip(*self.infos.values()):
            total.append(np.array(v).sum())
        total = np.array(total)
        t_mean = np.mean(total)
        t_median = np.median(total)

        for k, v in self.infos.items():
            av = np.array(v)
            mean = np.mean(av)
            median = np.median(av)
            self.infos[k] = [mean, median]

            find = False
            for c in self.collects:
                if c in k:
                    self.collect_infos[c].append(av)
                    find = True
                    break
            if not find:
                self.collect_infos["Others"].append(av)

            print(k)
        import prettytable
        table = prettytable.PrettyTable(["name", "layers", "average (ms)", "median (ms)", "percentage (%)"])
        table.set_style(prettytable.MARKDOWN)
        for k, v in self.collect_infos.items():
            av = np.array(v).sum(axis=0)
            mean = np.mean(av)
            median = np.median(av)
            table.add_row((k, len(v), "%.3f" % mean, "%.3f" % median, "%.1f" % (100 * median / t_median)))
        table.align = "r"
        table.add_row(("Total", sum(len(v) for v in self.collect_infos.values()),
                       "%.1f" % t_mean, "%.1f" % t_median, "%.1f" % 100))
        print(table)
