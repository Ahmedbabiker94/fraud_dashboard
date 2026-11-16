"""Streamlit dashboard for reproducible fraud research with the original PyTorch checkpoint."""
from pathlib import Path
import pandas as pd
import streamlit as st
from model_inference import load_checkpoint, predict_rows

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='Fraud Detection Research', layout='wide')
st.title('Credit Card Fraud Detection — Research Dashboard')
st.caption('Experimental graph model · batch-dependent inference · not a real-time banking product')

@st.cache_resource
def checkpoint():
    return load_checkpoint(ROOT/'model'/'fraud_model.pt')

@st.cache_data
def dataset(nrows):
    return pd.read_csv(ROOT/'data'/'creditcard_sample.csv',nrows=nrows)

try:
    artifact=checkpoint()
except Exception as e:
    st.error(f'Checkpoint unavailable/incompatible: {e}')
    st.stop()

st.subheader('Reported experiment results (embedded in trained checkpoint)')
st.warning('These figures were saved during training; they have not been independently reproduced in this dashboard.')
for phase in ('validation_metrics','test_metrics'):
    m=artifact.get(phase)
    if m:
        st.markdown('**'+phase.replace('_',' ').title()+'**')
        cols=st.columns(5)
        for c,(key,label) in zip(cols,[('precision','Precision'),('recall','Recall'),('f1','F1'),('roc_auc','ROC-AUC'),('pr_auc','PR-AUC')]):
            v=m.get(key)
            c.metric(label,f'{v:.3f}' if isinstance(v,(int,float)) else 'N/A')
        st.caption(f"Rows: {m.get('rows','?')}; fraud cases: {m.get('fraud_rows','?')}; decision threshold: {m.get('threshold','?')}")

st.divider()
st.subheader('Sample dataset and model inference')
st.info('The model is a graph/batch model: predictions depend on neighboring samples. Scores are not calibrated fraud probabilities. The subset below is NOT the original held-out test split.')
limit=st.slider('Rows to inspect (from sample dataset)',128,2048,512,128)
try:
    data=dataset(limit)
except Exception as e:
    st.error(f'Unable to read sample dataset: {e}')
    st.stop()
if 'Class' not in data.columns:
    st.error('Data requires a Class label for display')
    st.stop()
st.metric('Dataset rows loaded',len(data))
st.metric('Actual fraud cases',int(pd.to_numeric(data['Class'],errors='coerce').sum()))
run=st.button('Run saved hybrid model on displayed rows')
if run:
    try:
        pred=predict_rows(data,artifact)
        output=pd.concat([data.reset_index(drop=True),pred],axis=1)
        if pred['Anomaly score'].isna().any():
            st.warning('A final batch with fewer than three rows was excluded, matching notebook evaluation behavior.')
        st.metric('Predicted fraud',int((pred['Prediction']==1).sum()))
        st.bar_chart(pred['Prediction'].value_counts().sort_index())
        st.dataframe(output.head(100),use_container_width=True)
        st.download_button('Download inference CSV',output.to_csv(index=False),file_name='fraud_inference_sample.csv',mime='text/csv')
    except ModuleNotFoundError as e:
        st.error('Missing dependency: '+str(e)+'. Install requirements.txt in a supported Python environment.')
    except Exception as e:
        st.error('Inference failed: '+str(e))
