import os


class Config:
    def __init__(self, inputFolder, resultsFolder, graph_names, hedge_size, features, neg_types, imb_ratio, seed):

        self.task = "predict"
        self.input_folder = inputFolder
        self.results_folder = resultsFolder

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

        if type(features) != list:
            self.FEATURES = [features]
        else:
            self.FEATURES = features

        self.default_save_dir = os.path.join(self.input_folder, "{}", "hedge-{}", "negtype-{}-{}", "seed-{}")
        self.default_save_path = os.path.join(self.default_save_dir, "hypergraph.data")
        self.default_results_dir = os.path.join(self.results_folder, "hedge-{}", "negtype-{}-{}", "seed-{}")
        self.SEED = seed
