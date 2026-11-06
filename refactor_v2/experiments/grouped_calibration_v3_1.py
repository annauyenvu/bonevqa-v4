"""Frozen one-feature Platt calibration; caller must enforce data-role authorization."""
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

PARAMETERS=dict(C=1.0,solver='lbfgs',max_iter=1000,tol=1e-8,random_state=42,fit_intercept=True,class_weight=None)

def features(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():
        raise ValueError('Expected finite one-dimensional probabilities in [0,1]')
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

def fit(probabilities,labels):
    x=features(probabilities);y=np.asarray(labels)
    if y.ndim!=1 or len(y)!=len(x) or not np.isin(y,[0,1]).all():
        raise ValueError('Expected existing binary labels matching probability count')
    if len(np.unique(y))!=2:
        return {'passed':False,'failure':'single_class','parameters':PARAMETERS}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model=LogisticRegression(**PARAMETERS).fit(x,y)
    messages=[str(w.message) for w in caught]
    if any(issubclass(w.category,ConvergenceWarning) for w in caught):
        return {'passed':False,'failure':'convergence_warning','warnings':messages,'parameters':PARAMETERS}
    slope=float(model.coef_[0,0]);intercept=float(model.intercept_[0])
    if not np.isfinite([slope,intercept]).all():
        return {'passed':False,'failure':'nonfinite_parameters','warnings':messages,'parameters':PARAMETERS}
    return {'passed':True,'slope':slope,'intercept':intercept,'n_iter':int(model.n_iter_[0]),'warnings':messages,'parameters':PARAMETERS}

def apply(probabilities,calibration):
    if not calibration['passed']:raise ValueError('Cannot apply failed calibration')
    z=features(probabilities).ravel()*calibration['slope']+calibration['intercept']
    return np.exp(-np.logaddexp(0,-z))
