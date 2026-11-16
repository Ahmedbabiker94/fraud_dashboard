from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest
from model_inference import load_checkpoint,make_features

ROOT=Path(__file__).resolve().parents[1]

def test_checkpoint_schema():
    ck=load_checkpoint(ROOT/'model'/'fraud_model.pt')
    assert ck['model_config']['in_channels']==len(ck['feature_names'])==35
    assert 0<ck['threshold']<1
    assert ck['test_metrics']['rows']==3187
    assert ck['test_metrics']['fraud_rows']==15

def test_preprocessing_valid():
    ck=load_checkpoint(ROOT/'model'/'fraud_model.pt')
    df=pd.read_csv(ROOT/'data'/'creditcard_sample.csv',nrows=5)
    x=make_features(df,ck)
    assert x.shape==(5,35)
    assert pd.notna(x).all()

def test_missing_feature_rejected():
    ck=load_checkpoint(ROOT/'model'/'fraud_model.pt')
    df=pd.read_csv(ROOT/'data'/'creditcard_sample.csv',nrows=5).drop(columns=['V1'])
    with pytest.raises(ValueError,match='Missing raw columns'):
        make_features(df,ck)

def test_reported_metrics_consistent():
    ck=load_checkpoint(ROOT/'model'/'fraud_model.pt')
    test=ck['test_metrics']
    tp=11;fp=18;fn=4
    assert abs(test['precision']-tp/(tp+fp))<1e-10
    assert abs(test['recall']-tp/(tp+fn))<1e-10
    assert abs(test['f1']-2*tp/(2*tp+fp+fn))<1e-10
