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