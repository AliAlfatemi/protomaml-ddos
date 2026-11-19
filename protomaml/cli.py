"""CPU training and synthetic validation of the original episodic learner."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .model import ProtoMAMLNet
from .dataset import EpisodeDataset
from .train_and_eval import train_few_shot,evaluate_few_shot


def serialize(value):
    if isinstance(value,dict):return {k:serialize(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return serialize(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def train(csv,output,*,label='Label',benign_label='BENIGN',episodes=200,eval_episodes=100,
          k_shot=5,q_query=5,steps_inner=1,seed=42):
    import joblib
    output=Path(output)
    if output.exists():raise ValueError('output directory exists; choose a fresh run directory')
    if min(episodes,eval_episodes,k_shot,q_query,steps_inner)<1:raise ValueError('training counts must be positive')
    np.random.seed(seed);torch.manual_seed(seed)
    frame=pd.read_csv(csv);frame.columns=frame.columns.str.strip()
    if label not in frame:raise ValueError('missing label column: '+label)
    frame=frame.dropna(subset=[label]); y=(frame.pop(label).astype(str).str.strip()!=str(benign_label)).astype(int).to_numpy()
    frame=frame.drop(columns=['Flow ID','Source IP','Destination IP','Source Port','Destination Port','Timestamp','attack_cat','OrigLabel','Unnamed: 0'],errors='ignore')
    X=frame.apply(pd.to_numeric,errors='raise').replace([np.inf,-np.inf],np.nan)
    valid=X.notna().all(axis=1).to_numpy();X=X.to_numpy(dtype=np.float32)[valid];y=y[valid]
    tr,te=train_test_split(np.arange(len(y)),test_size=0.3,random_state=seed,stratify=y)
    scaler=StandardScaler();Xtr=scaler.fit_transform(X[tr]).astype(np.float32);Xte=scaler.transform(X[te]).astype(np.float32)
    data=EpisodeDataset(Xtr,y[tr],episodes=episodes,k_shot=k_shot,q_query=q_query)
    if any(np.sum(y[te]==c)<k_shot+q_query for c in (0,1)):raise ValueError('test classes need more distinct support/query rows')
    model=ProtoMAMLNet(X.shape[1]);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    losses,_=train_few_shot(model,DataLoader(data,batch_size=1),opt,steps_inner=steps_inner,episodes=episodes)
    metrics=serialize(evaluate_few_shot(model,Xte,y[te],steps_inner=steps_inner,episodes=eval_episodes,k_shot=k_shot,q_query=q_query))
    output.mkdir(parents=True)
    torch.save(model.state_dict(),output/'model.pt');joblib.dump(scaler,output/'scaler.joblib')
    meta={'seed':seed,'feature_columns':list(frame.columns),'input_dim':X.shape[1],
          'split_sizes':{'train':len(tr),'test':len(te)},'adaptation':'first-order copied query gradients',
          'support_query_sampling':'distinct rows without replacement','preprocessing_fit':'training rows only',
          'training_episodes':len(losses)}
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    return metrics


def smoke():
    from sklearn.datasets import make_classification
    X,y=make_classification(n_samples=200,n_features=6,n_informative=4,random_state=7)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'data.csv';f=pd.DataFrame(X);f['Label']=y;f.to_csv(p,index=False)
        r=train(p,Path(d)/'run',benign_label='0',episodes=3,eval_episodes=2,k_shot=2,q_query=2)
        assert 0<=r['accuracy']<=1
    return {'status':'passed','scope':'synthetic episodic software check'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('smoke');t=sub.add_parser('train')
    t.add_argument('--csv',required=True);t.add_argument('--output',required=True)
    t.add_argument('--label',default='Label');t.add_argument('--benign-label',default='BENIGN')
    for n,v in [('episodes',200),('eval-episodes',100),('k-shot',5),('q-query',5),('steps-inner',1),('seed',42)]:t.add_argument('--'+n,type=int,default=v)
    a=vars(p.parse_args(argv));c=a.pop('command')
    try:r=smoke() if c=='smoke' else train(**a)
    except (ValueError,FileNotFoundError) as e:p.error(str(e))
    print(json.dumps(r,indent=2,allow_nan=False))

if __name__=='__main__':main()
