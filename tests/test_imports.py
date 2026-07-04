def test_core_imports():
    import mne
    import moabb
    import sklearn
    import numpy
    import scipy
    import pandas
    import matplotlib

    assert mne is not None
    assert moabb is not None
    assert sklearn is not None
    assert numpy is not None
    assert scipy is not None
    assert pandas is not None
    assert matplotlib is not None


def test_project_imports():
    from bci_mi.config import FMAX, FMIN, SUBJECTS
    from bci_mi.data import load_left_right_data

    assert FMIN == 8
    assert FMAX == 30
    assert SUBJECTS == [1]
    assert load_left_right_data is not None
