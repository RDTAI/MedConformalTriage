import numpy as np
from medtriage import *

def test_class_conditional_sets_and_routes():
    p=np.array([[.9,.1],[.2,.8],[.55,.45],[.05,.95]])
    y=np.array([0,1,0,1]); q=fit_class_conditional(p,y,alpha=.5)
    sets=prediction_sets(p,q)
    result=route_cases(p,sets,ood_scores=np.array([-3,-3,2,-3]),ood_threshold=0,high_risk_classes={1},high_risk_min_confidence=.9)
    assert result.route[2] == REJECT
    assert result.route[3] == AUTO

def test_metrics_workload_sums_to_one():
    p=np.array([[.9,.1],[.5,.5],[.1,.9]]); sets=np.array([[1,0],[1,1],[0,1]],bool); y=np.array([0,1,0])
    result=route_cases(p,sets); report=evaluate_triage(result,y,groups=["a","a","b"])
    assert np.isclose(report["auto_rate"]+report["review_rate"]+report["reject_rate"],1)
    assert "worst_group_coverage" in report
