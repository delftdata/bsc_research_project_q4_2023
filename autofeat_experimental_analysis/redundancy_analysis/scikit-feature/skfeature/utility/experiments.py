from skfeature.information_theoretical_based import JMI, MIFS, CIFE, MRMR, CMIM

def select_mifs(X, y, n_selected_features):
    """
     Helper function to run MIFS feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return MIFS.mifs(X, y, n_selected_features=n_selected_features)


def select_mrmr(X, y, n_selected_features):
    """
     Helper function to run MRMR feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return MRMR.mrmr(X, y, n_selected_features=n_selected_features)


def select_jmi(X, y, n_selected_features):
    """
     Helper function to run JMI feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return JMI.jmi(X, y, n_selected_features=n_selected_features)


def select_cife(X, y, n_selected_features):
    """
     Helper function to run CIFE feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return CIFE.cife(X, y, n_selected_features=n_selected_features)


def select_cmim(X, y, n_selected_features):
    """
     Helper function to run CIFE feature selection.

    Args:
        X: training set without target column
        y: target column
        n_selected_features: number of features to select

    Returns:
        Result from feature selection
    """
    return CMIM.cmim(X, y, n_selected_features=n_selected_features)