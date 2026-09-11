import pandas as pd
from src.engine_v2.backtest.cpcv import make_folds, cpcv_combos, purged_train_index

def test_make_folds_covers_index():
    idx = pd.date_range("2007-01-01", "2020-12-31", freq="B")
    folds = make_folds(idx, n_folds=10)
    covered = pd.DatetimeIndex(sorted(set().union(*folds)))
    assert covered.equals(idx)

def test_cpcv_10_choose_2_is_45():
    combos = cpcv_combos(10, 2)
    assert len(combos) == 45
    assert all(isinstance(c, tuple) and len(c) == 2 for c in combos)

def test_purge_removes_neighboring_days():
    idx = pd.date_range("2020-01-01", "2020-03-31", freq="B")
    folds = make_folds(idx, n_folds=5)
    train = purged_train_index(idx, [folds[2]], embargo_days=5)
    test_min, test_max = folds[2].min(), folds[2].max()
    for d in train:
        # must not fall inside the purge/embargo window around fold 2
        assert not (test_min - pd.Timedelta(days=5) <= d <= test_max + pd.Timedelta(days=5))

def test_no_purge_when_embargo_zero():
    idx = pd.date_range("2020-01-01", "2020-06-30", freq="B")
    folds = make_folds(idx, n_folds=5)
    train = purged_train_index(idx, [folds[2]], embargo_days=0)
    assert not set(train).intersection(set(folds[2]))
    # everything except test fold is train
    non_test = pd.DatetimeIndex(sorted(set(idx) - set(folds[2])))
    assert train.equals(non_test)
