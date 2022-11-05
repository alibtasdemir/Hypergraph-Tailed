import os


class Config:
    def __init__(self, inputFolder, outFolder, graph_names, hedge_size, neg_types, imb_ratio, seed, save=False,
                 info=False):
        self.MAX_NEG_GENERATION = 10 * 1000 * 1000
        self.MAX_ITER = 10 * 1000 * 1000
        self.task = "create"
        self.input_folder = inputFolder
        self.output_folder = outFolder

        if type(hedge_size) == int:
            self.HEDGE_SIZES = [hedge_size]
        else:
            self.HEDGE_SIZES = hedge_size

        if type(imb_ratio) == int:
            self.IMBS = [imb_ratio]
        else:
            self.IMBS = imb_ratio

        if type(graph_names) != list:
            self.GNS = [graph_names]
        else:
            self.GNS = graph_names

        if type(neg_types) != list:
            self.NEG_TYPES = [neg_types]
        else:
            self.NEG_TYPES = neg_types

        self.SEED = seed
        self.default_save_dir = os.path.join(self.output_folder, "{}", "hedge-{}", "negtype-{}-{}", "seed-{}")
        self.default_save_path = os.path.join(self.default_save_dir, "hypergraph.data")
        self.save = save
        self.info = info
