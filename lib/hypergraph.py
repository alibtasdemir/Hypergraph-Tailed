import pickle
import random
import sys

import config
from lib.helper import *
from lib.sampler import get_cliques, get_hubs, get_tailed
import gc

class HyperGraph:
    def __init__(self, config, graph_name, hedge_size, neg_type, imb, seed):
        self.config = config

        self.gn = graph_name
        self.hedge_size = hedge_size
        self.neg_type = neg_type
        self.imb = imb
        self.seed = seed

        self.transactions, self.hedges, self.weights = get_hedges_and_weights(self.config, self.gn)

        # Will be Assigned after sample_hedges()
        # Train
        self.pos_hedges_train = []
        self.neg_hedges_train = []
        self.vis_hedges_train = []
        self.vis_weights_train = []
        # Will be Assigned after generate_pgs()
        self.pg2_train = None
        self.pg3_train = None
        self.pg4_train = None
        # Will be Assigned after sample_hedges()
        # Test
        self.pos_hedges_test = []
        self.neg_hedges_test = []
        self.vis_hedges_test = []
        self.vis_weights_test = []
        # Will be Assigned after generate_pgs()
        self.pg2_test = None
        self.pg3_test = None
        self.pg4_test = None

        self.sample_hedges()
        self.generate_pgs()

    @classmethod
    def from_pickle(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def sample_hedges(self):
        print("Sampling positive and negative hedges...")
        random.seed(self.seed)

        hw_dict = dict(zip(self.hedges, self.weights))
        # positive sampling
        pos_hedges, s_hedges, n_pos = self.__positive_sampling()

        # Negative sampling
        neg_hedges = self.__negative_sampling(s_hedges, n_pos)
        n_neg = len(neg_hedges)

        # Train/Test split
        pos_hedges_train = random.sample(pos_hedges, k=int(n_pos * 0.5))
        neg_hedges_train = random.sample(neg_hedges, k=int(n_neg * 0.5))

        pos_hedges_test = set(pos_hedges) - set(pos_hedges_train)
        neg_hedges_test = set(neg_hedges) - set(neg_hedges_train)

        # Visible hedges
        vis_hedges_train = set(self.hedges) - set(pos_hedges_train) - set(pos_hedges_test)
        vis_hedges_test = set(self.hedges) - set(pos_hedges_test)
        # Visible weights
        vis_weights_train = [hw_dict[hedge] for hedge in vis_hedges_train]
        vis_weights_test = [hw_dict[hedge] for hedge in vis_hedges_test]

        assert (set(pos_hedges_train).isdisjoint(neg_hedges_train))
        assert (set(pos_hedges_train).isdisjoint(vis_hedges_train))
        assert (set(pos_hedges_test).isdisjoint(neg_hedges_test))
        assert (set(pos_hedges_test).isdisjoint(vis_hedges_test))

        # Test train split
        self.pos_hedges_train = list(pos_hedges_train)
        self.neg_hedges_train = list(neg_hedges_train)
        self.vis_hedges_train = list(vis_hedges_train)
        self.vis_weights_train = list(vis_weights_train)

        self.pos_hedges_test = list(pos_hedges_test)
        self.neg_hedges_test = list(neg_hedges_test)
        self.vis_hedges_test = list(vis_hedges_test)
        self.vis_weights_test = list(vis_weights_test)

    def __positive_sampling(self, max_delete_prop=0.4):
        N = len(self.hedges)
        s_hedges = set()
        for idx, hedge in enumerate(self.hedges):
            if len(hedge) == self.hedge_size:
                s_hedges.add(hedge)

        n_s_hedges = len(s_hedges)
        max_pos = int(N * max_delete_prop)
        n_pos = min(max_pos, n_s_hedges)
        pos_hedges = random.sample(s_hedges, k=n_pos)
        return pos_hedges, s_hedges, n_pos

    def __negative_sampling(self, hedges_exclude, n_pos):
        pg_all = hedges_to_pg(self.hedges, self.weights, p=2)
        max_neg = int(n_pos * self.imb)
        n_neg = min(max_neg, self.config.MAX_NEG_GENERATION)

        if self.neg_type in ["hub", "star"]:
            neg_hedges = get_hubs(self.hedge_size, G=pg_all, exclude=hedges_exclude, n=n_neg, induced=False, max_iter=self.config.MAX_ITER)
        elif self.neg_type == "clique":
            neg_hedges = get_cliques(self.hedge_size, G=pg_all, exclude=hedges_exclude, n=n_neg, strong=False, max_iter=self.config.MAX_ITER)
        elif self.neg_type == "tailed":
            neg_hedges = get_tailed(self.hedge_size, G=pg_all, exclude=hedges_exclude, n=n_neg, max_iter=self.config.MAX_ITER)
        else:
            raise NotImplementedError

        return neg_hedges

    def generate_pgs(self):
        print("Generating pgs..")

        assert (len(self.vis_hedges_train) != 0)
        assert (len(self.vis_weights_train) != 0)
        assert (len(self.vis_hedges_test) != 0)
        assert (len(self.vis_weights_test) != 0)

        self.pg2_train = hedges_to_pg(self.vis_hedges_train, self.vis_weights_train, p=2)
        self.pg2_test = hedges_to_pg(self.vis_hedges_test, self.vis_weights_test, p=2)

        self.pg3_train = hedges_to_pg(self.vis_hedges_train, self.vis_weights_train, p=3)
        self.pg3_test = hedges_to_pg(self.vis_hedges_test, self.vis_weights_test, p=3)

        if self.hedge_size > 4 or self.hedge_size is None:
            self.pg4_train = hedges_to_pg(self.vis_hedges_train, self.vis_weights_train, p=4)
            self.pg4_test = hedges_to_pg(self.vis_hedges_test, self.vis_weights_test, p=4)
        print("..done")

    def __str__(self):
        out = ""
        out += ("-" * 30) + "\n"
        out += "HyperGraph: {}\n".format(self.gn)
        out += "Target Hedge Size: {}\n".format(self.hedge_size)

        out += "# of transactions: {}\n".format(len(self.transactions))
        out += "# of hedges: {}\n".format(len(self.hedges))
        out += ("-" * 30) + "\n"

        out += "Pos/Neg samples\n"
        out += "Settings:\nSampling type: {}\nSeed: {}\tImbalance: {}\n".format(self.neg_type, self.seed, self.imb)
        out += ("-" * 30) + "\n"
        out += "Train:\n"
        out += "Neg: {}\tPos: {}\nImbalance: {:.2f} (Actual)\n".format(
            len(self.neg_hedges_train),
            len(self.pos_hedges_train),
            len(self.neg_hedges_train) / len(self.pos_hedges_train)
        )
        out += ("-" * 30) + "\n"
        out += "Test:\n"
        out += "Neg: {}\tPos: {}\nImbalance: {:.2f} (Actual)\n".format(
            len(self.neg_hedges_test),
            len(self.pos_hedges_test),
            len(self.neg_hedges_test) / len(self.pos_hedges_test)
        )
        out += ("-" * 30) + "\n"
        return out

    def save_data(self, name=None):
        # '../../hg-MAX%s/{}/size-{}/neg-{}{}/seed-{}/' % MAX_HEDGE_SIZE
        # (gn, hedge_size, neg_type, imb, seed)
        # 'HG\\{}\\hedge-{}\\negtype-{}-{}\\seed-{}\\hypergraph.data'
        folder_path = self.config.default_save_dir.format(self.gn, self.hedge_size, self.neg_type, self.imb, self.seed)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if name:
            save_path = os.path.join(folder_path, "{}_hypergraph.data".format(name))
        else:
            save_path = os.path.join(folder_path, "hypergraph.data")
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def print_summary(self):
        print("------------------------------------")
        print("Summary of HyperGraph")
        print("\n")

        print("Task")
        print("graph name: {}".format(self.gn))
        print("target hedge size: {}".format(self.hedge_size))
        print("\n")

        print("Pos/Neg")
        print("neg: {}{}".format(self.neg_type, self.imb))
        print("seed: {}".format(self.seed))
        print("\n")

        print("All")
        print("# transactions: {}".format(len(self.transactions)))
        print("# hedges: {}".format(len(self.hedges)))
        print("\n")

        print("Train")
        print("# vis hedges: {}".format(len(self.vis_hedges_train)))
        print("# pos hedges: {}".format(len(self.pos_hedges_train)))
        print("# neg hedges: {}".format(len(self.neg_hedges_train)))
        print("pos/neg: {}".format(len(self.pos_hedges_train) / float(len(self.neg_hedges_train))))
        print("\n")

        print("Test")
        print("# vis hedges: {}".format(len(self.vis_hedges_test)))
        print("# pos hedges: {}".format(len(self.pos_hedges_test)))
        print("# neg hedges: {}".format(len(self.neg_hedges_test)))
        print("pos/neg: {}".format(len(self.pos_hedges_test) / float(len(self.neg_hedges_test))))
        print("\n")


def generate_hypergraphs(config):
    import time
    for gn in config.GNS:
        for hs in config.HEDGE_SIZES:
            for neg_type in config.NEG_TYPES:
                for imb in config.IMBS:
                    print(
                        "#" * 30 + "\n"
                        + "Settings:\n"
                        + "Graph Name: {}\n".format(gn)
                        + "Hyperedge Size: {}\n".format(hs)
                        + "Sampling Type: {}\n".format(neg_type)
                        + "Imbalance Ratio: {}\n".format(imb)
                        + "#" * 30
                    )
                    gc.collect()

                    filename = config.default_save_path.format(gn, hs, neg_type, imb, config.SEED)
                    start_time = time.time()
                    hg = HyperGraph(config, gn, hs, neg_type, imb, config.SEED)
                    if config.save:
                        print("Saving...")
                        hg.save_data()
                        print("Saved!")
                    if config.info:
                        print(hg)

                    print("--- {:0.3f} seconds ---\n".format(time.time() - start_time))


def run(config):
    from lib import Logger

    sys.stdout = Logger.Logger(config.task)
    generate_hypergraphs(config)
    sys.stdout.close()


if __name__ == "__main__":
    from lib import Logger
    import os
    import time
    from config.create_config import Config

    input_folder = "../proc_data"
    output_folder = "HG_NEW"
    graph_names = "DAWN"
    hedge_size = 4
    neg_types = "clique"
    imb_ratio = 1

    config = Config(
        input_folder,
        output_folder,
        graph_names,
        hedge_size,
        neg_types,
        imb_ratio,
        seed=9,
        save=True,
        info=True
    )

    sys.stdout = Logger.Logger()
    generate_hypergraphs(config)
    sys.stdout.close()
    """
    from lib import Logger

    sys.stdout = Logger.Logger()
    iterate_over_data(
        GNS=["DAWN"],
        HEDGE_SIZES=[4, 5],
        NEG_TYPES=["clique", "star", "tailed"],
        IMBS=[5, 10]
    )
    #complete_pgs()
    sys.stdout.close()
    """

    """
    GN = "contact-primary-school"
    hedge_size = 4
    imbalance = 5
    neg_type = "star"
    seed = 9
    filename = default_save_path.format(GN, hedge_size, neg_type, imbalance, seed)

    # graph_name, hedge_size, neg_type, imb, seed
    hg = HyperGraph(GN, hedge_size, neg_type, imbalance, seed)
    hg.save_data()
    #hg = HyperGraph.from_pickle(filename)
    print(hg)
    """
