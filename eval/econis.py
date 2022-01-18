"""
Executable to run AAE on the Econis dataset
"""
import argparse
import json
import re

from gensim.models.keyedvectors import KeyedVectors

from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import Countbased
from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.svd import SVDRecommender
from aaerec.vae import VAERecommender
from aaerec.dae import DAERecommender
from aaerec.condition import ConditionList, PretrainedWordEmbeddingCondition, CategoricalCondition

def log(*print_args, logfile=None):
    """ Maybe logs the output also in the file `outfile` """
    if logfile:
        with open(logfile, 'a') as fhandle:
            print(*print_args, file=fhandle)
    print(*print_args)


# Set to a folder containing the EconBiz (extended) dataset

DATA_PATH = "../econis/econbiz62k-extended.json"
DEBUG_LIMIT = None
METRICS = ['mrr', 'map']

if __name__ == "__main__":
    # Set to the word2vec-Google-News-corpus file
    W2V_PATH = "../vectors/GoogleNews-vectors-negative300.bin.gz"
    W2V_IS_BINARY = True
    print("Loading pre-trained embedding", W2V_PATH)
    VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

    # Hyperparameters
    ae_params = {
        'n_code': 50,
        'n_epochs': 100,
        # 'embedding': VECTORS,
        'batch_size': 1000,
        'n_hidden': 100,
        'normalize_inputs': True,
    }
    vae_params = {
        'n_code': 50,
        # VAE results get worse with more epochs in preliminary optimization
        #(Pumed with threshold 50)
        'n_epochs': 50,
        'batch_size': 1000,
        'n_hidden': 100,
        'normalize_inputs': True,
    }


    # Models without metadata
    BASELINES = [
        # RandomBaseline(),
        # MostPopular(),
        Countbased(),
        SVDRecommender(1000, use_title=False),
    ]
    RECOMMENDERS = [
        AAERecommender(adversarial=False,
                    conditions=None,
                    lr=0.001,
                    **ae_params),
        AAERecommender(adversarial=True,
                    conditions=None,
                    gen_lr=0.001,
                    reg_lr=0.001,
                    **ae_params),
        VAERecommender(conditions=None, **vae_params),
        DAERecommender(conditions=None, **ae_params)
    ]

    # Metadata to use
    CONDITIONS = ConditionList([
        ('title', PretrainedWordEmbeddingCondition(VECTORS)),
        ('author', CategoricalCondition(embedding_dim=32, reduce="sum",
                                        sparse=True, embedding_on_gpu=True))
    ])

    # Model with metadata (metadata used as set in CONDITIONS above)
    CONDITIONED_MODELS = [
        # TODO SVD can use only titles not generic conditions
        # SVDRecommender(1000, use_title=True),
        AAERecommender(adversarial=False,
                    conditions=CONDITIONS,
                    lr=0.001,
                    **ae_params),
        AAERecommender(adversarial=True,
                    conditions=CONDITIONS,
                    gen_lr=0.001,
                    reg_lr=0.001,
                    **ae_params),
        DecodingRecommender(CONDITIONS,
                        n_epochs=100, batch_size=1000, optimizer='adam',
                        n_hidden=100, lr=0.001, verbose=True),
        VAERecommender(conditions=CONDITIONS, **vae_params),
        DAERecommender(conditions=CONDITIONS, **ae_params)
    ]


def load(path):
    """ Loads a single file """
    with open(path, 'r') as fhandle:
        obj = json.load(fhandle)
    return obj


def parse_en_labels(subjects):
    """
    From subjects in the json formats to a list of english descriptors of subjects
    """
    labels = []
    for subject in subjects:
        if subject["name_en"] != "":
            labels.append(subject["name_en"])

    return labels


def parse_authors(p):
    """
    From Marc21-IDs in the json formats to a list of authors
    """
    authors = []
    try:
        for creator in p.pop("creator_personal"):
            authors.append(creator.pop("name"))
    except KeyError:
        pass

    try:
        for contributor in p.pop("contributor_personal"):
            authors.append(contributor.pop("name"))
    except KeyError:
        pass

    return authors


def unpack_papers_conditions(papers):
    """
    Unpacks list of papers in a way that is compatible with our Bags dataset
    format. It is not mandatory that papers are sorted.
    """

    bags_of_labels, ids, side_info, years, authors = [], [], {}, {}, {}
    subjects_cnt, title_cnt, authors_cnt = 0, 0, 0
    for paper in papers:
        # Extract ids
        ids.append(paper["econbiz_id"])
        # Put all subjects assigned to the paper in here
        try:
            # Subject may be missing
            subjects = parse_en_labels(paper["subject_stw"])
            bags_of_labels.append(subjects)
            if len(subjects) > 0:
                subjects_cnt += 1
        except KeyError:
            bags_of_labels.append([])

        # Use dict here such that we can also deal with unsorted ids
        try:
            side_info[paper["econbiz_id"]] = paper["title"]
            if paper["title"] != "":
                title_cnt += 1
        except KeyError:
            side_info[paper["econbiz_id"]] = ""
        try:
            # Sometimes data in format yyyy.mm.dd (usually only year)
            if type(paper["date"]) is str:
                paper["date"] = int(paper["date"][:4])
            years[paper["econbiz_id"]] = paper["date"]
        except KeyError:
            years[paper["econbiz_id"]] = -1

        authors[paper["econbiz_id"]] = parse_authors(paper)
        if len(authors[paper["econbiz_id"]]) > 0:
            authors_cnt += 1

    print("Metadata-fields' frequencies: subjects={}, title={}, authors={}"
          .format(subjects_cnt / len(papers), title_cnt / len(papers), authors_cnt / len(papers)))

    # bag_of_labels and ids should have corresponding indices
    # In side_info the id is the key
    # Re-use 'title' and year here because methods rely on it
    return bags_of_labels, ids, {"title": side_info, "year": years, "author": authors}


def main(year, min_count=None, outfile=None, drop=1):
    """ Main function for training and evaluating AAE methods on IREON data """
    print("Loading data from", DATA_PATH)
    papers = load(DATA_PATH)
    print("Unpacking data...")
    # bags_of_papers, ids, side_info = unpack_papers(papers)
    bags_of_papers, ids, side_info = unpack_papers_conditions(papers)
    del papers
    bags = Bags(bags_of_papers, ids, side_info)
    if args.compute_mi:
        from aaerec.utils import compute_mutual_info
        print("[MI] Dataset: ECONIS")
        print("[MI] min Count:", min_count)
        tmp = bags.build_vocab(min_count=min_count, max_features=None)
        mi = compute_mutual_info(tmp, conditions=None, include_labels=True,
                                 normalize=True)
        with open('mi.csv', 'a') as mifile:
            print('EconBiz', min_count, mi, sep=',', file=mifile)
        print("=" * 78)
        exit(0)

    log("Whole dataset:", logfile=outfile)
    log(bags, logfile=outfile)

    evaluation = Evaluation(bags, year, logfile=outfile)
    evaluation.setup(min_count=min_count, min_elements=2, drop=drop)

    # Use only partial citations/labels list (no additional metadata)
    # with open(outfile, 'a') as fh:
    #     print("~ Partial List", "~" * 42, file=fh)
    # evaluation(BASELINES + RECOMMENDERS)

    # Use additional metadata (as defined in CONDITIONS for all models but SVD, which uses only titles)
    with open(outfile, 'a') as fh:
        print("~ Conditioned Models", "~" * 42, file=fh)
    evaluation(CONDITIONED_MODELS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('year', type=int,
                        help='First year of the testing set.')
    parser.add_argument('-m', '--min-count', type=int,
                        help='Pruning parameter', default=None)
    parser.add_argument('-o', '--outfile',
                        help="File to store the results.",
                        type=str, default=None)
    parser.add_argument('-dr', '--drop', type=str,
                        help='Drop parameter', default="1")
    parser.add_argument('--compute-mi', default=False,
                        action='store_true')
    args = parser.parse_args()

    # Drop could also be a callable according to evaluation.py but not managed as input parameter
    try:
        drop = int(args.drop)
    except ValueError:
        drop = float(args.drop)

    main(year=args.year, min_count=args.min_count, outfile=args.outfile, drop=drop)
