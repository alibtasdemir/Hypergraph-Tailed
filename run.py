import argparse
import sys, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='HyperGraphLink'
    )

    parser.add_argument(
        '--task',
        default="create",
        help="Choose the task. "
             "Options: "
             "'preproc' To preprocess raw files, "
             "'create' To create hypergraph data, "
             "'predict' To make predictions",
        choices=['preproc', 'create', 'predict']

    )

    parser.add_argument('--hgdir', type=str, default="HG", help='The folder with preprocessed hypergraphs')
    parser.add_argument('--resultsdir', type=str, default="Results", help="The folder where the results will be stored")
    parser.add_argument('--logdir', type=str, default="LOGS", help="The folder where the runtime logs will be stored")

    args, _ = parser.parse_known_args()

    if args.task == "preproc":
        parser.add_argument('--rawdir', type=str, default="data", help="The main folder with raw hypergraphs")
        parser.add_argument(
            '--outdir',
            type=str,
            default="data",
            help="The main folder with processed hypergraphs will be stored"
        )
        args, _ = parser.parse_known_args()
        from config.preproc_config import Config
        from lib.data_preproc import run_preproc

        config = Config(args.rawdir, args.outdir)

        from lib.Logger import Logger
        sys.stdout = Logger(task=args.task)
        run_preproc(config)
        sys.stdout.close()
        sys.exit()

    if args.task == "create":
        parser.add_argument(
            '--graphdir',
            type=str,
            default="data",
            help="The main folder with processed hypergraphs is stored"
        )

        parser.add_argument(
            '--graphname',
            type=str,
            default="email-Enron",
            help="The name of the graphname (Empty if you want to process all graphs in folder"
        )

        parser.add_argument(
            '--hedgesize',
            type=int,
            default=0,
            choices=[0, 4, 5],
            help='The size of the hyperedge. (0 for both 4, 5)'
        )

        parser.add_argument(
            '--sampling',
            type=str,
            default="",
            choices=["clique", "star", "tailed", ""],
            help='The negative sampling technique. (Empty for all)'
        )

        parser.add_argument(
            '--imb',
            type=int,
            default=0,
            choices=[0, 1, 2, 5, 10],
            help='Imbalance ratio for negative samples. (0 for all)'
        )

        args, _ = parser.parse_known_args()
        input_folder = args.graphdir
        output_folder = args.hgdir

        if args.graphname == "":
            graph_names = next(os.walk(input_folder))[1]
        else:
            if not os.path.exists(os.path.join(input_folder, args.graphname)):
                print("Graph not found at {}".format(os.path.join(input_folder, args.graphname)))
                sys.exit()
            graph_names = args.graphname

        if args.hedgesize == 0:
            hedge_size = [4, 5]
        else:
            hedge_size = args.hedgesize

        if args.sampling == "":
            neg_types = ["clique", "star", "tailed"]
        else:
            neg_types = args.sampling

        if args.imb == 0:
            imb_ratio = [1, 2, 5, 10]
        else:
            imb_ratio = args.imb

        from config.create_config import Config
        from lib.hypergraph import run

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

        run(config)
        sys.exit()

    if args.task == "predict":
        from config.prediction_config import Config

        parser.add_argument(
            '--graphname',
            type=str,
            default="email-Enron",
            help="The name of the graphname (Empty if you want to process all graphs in folder)"
        )

        parser.add_argument(
            '--hedgesize',
            type=int,
            default=0,
            choices=[0, 4, 5],
            help='The size of the hyperedge. (0 for both 4, 5)'
        )

        parser.add_argument(
            '--sampling',
            type=str,
            default="",
            choices=["clique", "star", "tailed", ""],
            help='The negative sampling technique. (Empty for all)'
        )

        parser.add_argument(
            '--imb',
            type=int,
            default=0,
            choices=[0, 1, 2, 5, 10],
            help='Imbalance ratio for negative samples. (0 for all)'
        )

        parser.add_argument(
            '--feature',
            type=str,
            default="",
            choices=["", "gm", "hm", "am", "cn", "jc", "aa"],
            help='Used features for prediction. (Empty for all)'
        )

        args, _ = parser.parse_known_args()
        input_folder = args.hgdir
        results_dir = args.resultsdir

        if args.graphname == "":
            graph_names = next(os.walk(input_folder))[1]
        else:
            if not os.path.exists(os.path.join(input_folder, args.graphname)):
                print("Graph not found at {}".format(os.path.join(input_folder, args.graphname)))
                sys.exit()
            graph_names = args.graphname

        if args.hedgesize == 0:
            hedge_size = [4, 5]
        else:
            hedge_size = args.hedgesize

        if args.sampling == "":
            neg_types = ["clique", "star", "tailed"]
        else:
            neg_types = args.sampling

        if args.imb == 0:
            imb_ratio = [1, 2, 5, 10]
        else:
            imb_ratio = args.imb

        if args.feature == "":
            features = ["gm", "hm", "am", "cn", "jc", "aa"]
        else:
            features = args.feature

        config = Config(
            input_folder,
            results_dir,
            graph_names,
            hedge_size,
            features,
            neg_types,
            imb_ratio,
            seed=9,
        )

        from lib.prediction import run_prediction

        run_prediction(config)
        sys.exit()
