import os

from mimikit.models.freqnet import FreqNet
from mimikit.models.sample_rnn import SampleRNN

from mimikit import NeptuneConnector

NC = NeptuneConnector(user="k-tonal",
                      setup={"project": "experiment-Stimme"})


proj = NC.get_project("project")
exps = proj.get_experiments()

all_experiments_by_id = {e.id: e for e in exps}

params = {e.id: e.get_parameters() for e in exps}

classes = dict(SampleRNN=SampleRNN, FreqNet=FreqNet)

model_classes = {exp_id: classes[p["model_class"]] for exp_id, p in params.items()}
files = {exp_id: eval(p["files"]) for exp_id, p in params.items()}


def download_states(destination, *exp_ids):
    """
    downloads the states of the experiment ids passed as *args

    Parameters
    ----------
    destination : where to download the states
    exp_ids: ids of the experiment to download the states from

    Examples
    --------
    # when in a colab notebook
    >>> download_states("/content/", "EXS-24", "EXS-28")
    """
    nc = NeptuneConnector(user="k-tonal", setup={})
    for eid in exp_ids:
        nc.setup[eid] = "experiment-Stimme/" + eid
        nc.download_experiment(eid, destination=destination, artifacts="states")


def load_exp_id_at_epochs(root_dir, *ids_and_epochs):
    """
    loads checkpoints located in `root_dir` for the experiment ids and epochs passed as *args.
    note that the dbs for the models are built if they don't exist yet.

    Parameters
    ----------
    root_dir: where the states are located
    ids_and_epochs: tuple of (str, int) e.g. ("EXS-21", 25)

    Returns
    -------
    list of loaded models

    Examples
    --------
    # when in a colab notebook
    >>> models = load_exp_id_at_epochs("/content/", ("EXS-23", 25), ("EXS-23", 50))
    """
    models = []
    for i, ep in ids_and_epochs:
        cls, exp_files = model_classes[i], files[i]
        hp = params[i]
        db_path = os.path.join(root_dir, i + ".h5")
        if cls is FreqNet:
            if not os.path.exists(db_path):
                db = cls.db_class.make(db_path, files=["./data/" + file for file in exp_files],
                                       n_fft=eval(hp["n_fft"]), hop_length=eval(hp["hop_length"]))
            else:
                db = cls.db_class(db_path)
            models += [FreqNet.load_from_checkpoint(os.path.join(root_dir, i, "states", "epoch=%i.ckpt" % ep), db=db)]
        elif cls is SampleRNN:
            if not os.path.exists(db_path):
                db = cls.db_class.make(db_path, files=["./data/" + file for file in exp_files], mu=hp["mu"])
            else:
                db = cls.db_class(db_path)
            models += [SampleRNN.load_from_checkpoint(os.path.join(root_dir, i, "states", "epoch=%i.ckpt" % ep), db=db)]

    return models