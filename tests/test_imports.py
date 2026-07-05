def test_core_imports():
    import matplotlib
    import mne
    import moabb
    import numpy
    import pandas
    import scipy
    import sklearn

    assert mne is not None
    assert moabb is not None
    assert sklearn is not None
    assert numpy is not None
    assert scipy is not None
    assert pandas is not None
    assert matplotlib is not None


def test_project_imports():
    from bci_mi.config import ALL_SUBJECTS, FMAX, FMIN, N_CSP_COMPONENTS, SUBJECTS
    from bci_mi.data import load_left_right_data
    from bci_mi.train import build_pipeline

    assert FMIN == 8
    assert FMAX == 30
    assert SUBJECTS == [1]
    assert ALL_SUBJECTS == list(range(1, 10))
    assert N_CSP_COMPONENTS == 6
    assert load_left_right_data is not None
    assert build_pipeline is not None


def test_pipeline_steps():
    from bci_mi.train import build_pipeline

    pipeline = build_pipeline()
    step_names = [name for name, _ in pipeline.steps]

    assert step_names == ["csp", "lda"]


def test_plotting_imports():
    from bci_mi.plot_results import plot_confusion_matrix

    assert plot_confusion_matrix is not None


def test_benchmark_imports():
    from bci_mi.benchmark import evaluate_subject, run_all_subjects

    assert evaluate_subject is not None
    assert run_all_subjects is not None

def test_benchmark_plotting_imports():
    from bci_mi.plot_benchmark import plot_all_subject_accuracies

    assert plot_all_subject_accuracies is not None