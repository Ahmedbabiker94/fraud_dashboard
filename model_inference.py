"""Checkpoint-backed batch inference for the ORIGINAL notebook architecture.

Do not use the checkpoint's stored test metrics as proof of live predictions.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch

EXPECTED_RAW = ['Time'] + [f'V{i}' for i in range(1,29)] + ['Amount']
BATCH_SIZE = 128  # identical to collect_scores() in training notebook


def load_checkpoint(path):
    # weights_only=True: never execute arbitrary pickle payloads through torch.load
    artifact=torch.load(path,map_location='cpu',weights_only=True)
    required={'model_state_dict','model_config','feature_names','amount_index',
              'robust_center','robust_scale','standard_mean','standard_scale','threshold'}
    if not isinstance(artifact,dict) or not required.issubset(artifact):
        raise ValueError('Incompatible checkpoint: missing model/preprocessing metadata')
    if len(artifact['standard_mean']) != len(artifact['feature_names']):
        raise ValueError('Incorrect preprocessing feature lengths')
    return artifact


def make_features(df,artifact):
    missing=set(EXPECTED_RAW)-set(df.columns)
    if missing: raise ValueError('Missing raw columns: '+', '.join(sorted(missing)))
    frame=df[EXPECTED_RAW].copy()
    for name in EXPECTED_RAW:
        frame[name]=pd.to_numeric(frame[name],errors='coerce')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Data contains NaN/inf or non-numeric features')
    frame['NormTime']=frame['Time']
    hour=(frame['Time']/3600)%24
    frame['Hour_sin']=np.sin(2*np.pi*hour/24)
    frame['Hour_cos']=np.cos(2*np.pi*hour/24)
    frame['V1_V3']=frame['V1']*frame['V3']
    frame['V4_V12']=frame['V4']*frame['V12']
    frame['V14_V17']=frame['V14']*frame['V17']
    frame=frame.drop(columns=['Time'])
    names=artifact['feature_names']
    if set(names)!=set(frame.columns):
        raise ValueError('Feature schema mismatch')
    x=frame[names].to_numpy(dtype=np.float32,copy=True)
    amount_index=int(artifact['amount_index'])
    x[:,amount_index]=(x[:,amount_index]-float(artifact['robust_center'][0]))/float(artifact['robust_scale'][0])
    mean=np.asarray(artifact['standard_mean'],dtype=np.float32)
    scale=np.asarray(artifact['standard_scale'],dtype=np.float32)
    if np.any(scale==0): raise ValueError('Invalid scaler metadata')
    return ((x-mean)/scale).astype(np.float32)


def predict_rows(frame,artifact,batch_size=BATCH_SIZE):
    from hybrid_model import EnhancedHybridGNNVAEGRU,create_graph_from_batch,compute_anomaly_scores
    if batch_size<3: raise ValueError('Batch size must be >=3')
    X=make_features(frame,artifact)
    model=EnhancedHybridGNNVAEGRU(**artifact['model_config'])
    model.load_state_dict(artifact['model_state_dict'],strict=True)
    model.eval()
    scores=np.full(len(X),np.nan,dtype=float)
    with torch.inference_mode():
        for start in range(0,len(X),batch_size):
            end=min(start+batch_size,len(X))
            if end-start<3: continue # matching notebook collect_scores behavior
            graph,_=create_graph_from_batch(X[start:end],np.zeros(end-start,dtype=int))
            main,aux,recon,_,mu,logvar=model(graph.x,graph.edge_index,graph.edge_attr)
            reconstruction=torch.mean((recon-graph.x).pow(2),dim=1)
            score=np.atleast_1d(compute_anomaly_scores(main,aux,reconstruction,mu,logvar))
            scores[start:end]=score
    return pd.DataFrame({'Anomaly score':scores,'Prediction':pd.array(np.where(np.isnan(scores),-1,(scores>=float(artifact['threshold'])).astype(int)),dtype='Int64')})
