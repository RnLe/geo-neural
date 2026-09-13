"""Explicit train/bake experiments. Baking is NOT a runtime neural decoder."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from geoneural.common import environment, read_json, sha_file, utc, write_json


def features(flat: np.ndarray, side: int, intervals: int, shared: bool) -> tuple[np.ndarray,np.ndarray]:
    row,col=flat//side,flat%side
    per_side=(side-1)//intervals
    tx=np.minimum(col//intervals,per_side-1)
    ty=np.minimum(row//intervals,per_side-1)
    tiles=(ty*per_side+tx).astype(np.int64)
    if shared:
        coords=np.column_stack([(col-tx*intervals)/intervals,(row-ty*intervals)/intervals])*2-1
    else:
        coords=np.column_stack([col/(side-1),row/(side-1)])*2-1
    return coords.astype(np.float32),tiles


def train(atlas: Path,out: Path,kind: str='siren',steps: int=2000,batch: int=8192,width: int=128,depth: int=3,seed: int=1729,device: str='cpu',checkpoint_every: int=500,extrapolation_fraction: float=0.25) -> Path:
    import torch
    from safetensors.torch import save_file
    from geoneural.neural.models import make_model
    from geoneural.neural import splits
    if out.exists(): raise FileExistsError('Use a fresh checkpoint directory')
    # Claim the directory before training rather than after, so a failure at the
    # end (including another process taking the path meanwhile) cannot discard
    # the whole run.
    out.mkdir(parents=True)
    if not 1<=steps<=1000000 or not 1<=batch<=1048576: raise ValueError('Invalid training work limits')
    m=read_json(atlas); refpath=atlas.parent/'reference.npy'
    if sha_file(refpath)!=m['reference_sha256']: raise ValueError('Reference changed')
    reference=np.load(refpath,mmap_mode='r',allow_pickle=False)
    if reference.size>16777216: raise ValueError('Regional prototype limited to 16M samples; implement streaming region datasets before expanding')
    side=reference.shape[0]; intervals=m['page_intervals']
    # Geographic holdouts, not random pixels. A pixel withheld from a 10 m
    # terrain field sits between eight neighbours that were not, so predicting
    # it is interpolation between known values and says nothing about transfer.
    split=splits.build(side,intervals,extrapolation_fraction)
    train_indexes=np.flatnonzero(split['trainMask'].reshape(-1))
    if train_indexes.size<batch: raise ValueError('training region is smaller than one batch')
    config={'kind':kind,'width':width,'depth':depth,'tiles':((side-1)//intervals)**2,'latent':16}
    torch.manual_seed(seed)
    # Separate streams for training batches and evaluation subsampling. Sharing
    # one generator would make the evaluation cadence part of the training
    # trajectory: changing checkpoint_every would draw different batches and
    # produce a different model.
    rng=np.random.default_rng(seed); eval_rng=np.random.default_rng(seed^0x5eed)
    model=make_model(config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    mean=float(np.mean(reference,dtype=np.float64)); scale=max(float(np.std(reference,dtype=np.float64)),1.0)
    def evaluate(mask_name: str, limit: int=131072) -> dict:
        """Full-reference error on a held-out region, reported with p99 and max.

        Mean error alone hides the failure that matters: a model can fit most of
        a region and miss a ridge, and the ridge is what a renderer shows.
        """
        pool=np.flatnonzero(split[mask_name].reshape(-1))
        if pool.size==0: return {'samples':0}
        picked=pool if pool.size<=limit else eval_rng.choice(pool,limit,replace=False)
        with torch.inference_mode():
            coords,tiles=features(picked,side,intervals,kind=='shared')
            predicted=model(torch.from_numpy(coords).to(device),torch.from_numpy(tiles).to(device)).squeeze(-1)
            errors=np.abs(predicted.detach().cpu().numpy().astype(np.float64)*scale+mean-reference.reshape(-1)[picked])
        return {'samples':int(picked.size),'mae_m':float(errors.mean()),'rmse_m':float(np.sqrt(np.mean(errors**2))),'p99_m':float(np.quantile(errors,.99)),'max_m':float(errors.max())}

    def persist(step: int, elapsed: float, history: list) -> None:
        """Write weights and metadata now, so an interrupted run keeps its work."""
        checkpoint=out/'weights.safetensors'
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(checkpoint))
        meta={'schema':'geoneural-coordinate-checkpoint-v1','model':config,'mean_m':mean,'scale_m':scale,'reference_side':side,'page_intervals':intervals,'source_atlas_id':m['content_id'],'reference_sha256':m['reference_sha256'],'weights_sha256':sha_file(checkpoint),'weights_bytes':checkpoint.stat().st_size,'steps_completed':step+1,'steps_requested':steps,'batch':batch,'seed':seed,'device':device,'training_seconds':elapsed,'split':splits.summary(split),'environment':environment(),'history':history,'created_utc':utc(),'evaluation_scope':'trained on the training pages only; interpolation and extrapolation errors are on withheld pages of the same region and are not evidence about a different landscape','qualification':'unqualified regional control; no residual coding, hard error bound, boundary guarantee or rate optimization'}
        write_json(out/'model.json',meta)

    history=[]; start=time.perf_counter()
    for step in range(steps):
        indexes=rng.choice(train_indexes,batch,replace=False)
        coords,tiles=features(indexes,side,intervals,kind=='shared')
        target=((reference.reshape(-1)[indexes]-mean)/scale).astype(np.float32)
        predicted=model(torch.from_numpy(coords).to(device),torch.from_numpy(tiles).to(device)).squeeze(-1)
        truth=torch.from_numpy(target).to(device)
        loss=(predicted-truth).square().mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step%100==0 or step==steps-1:
            point={'step':step,'batch_rmse_m':float(loss.detach().cpu().sqrt())*scale}
            if step%max(checkpoint_every,1)==0 or step==steps-1:
                model.eval()
                point['interpolation']=evaluate('interpolationMask')
                point['extrapolation']=evaluate('extrapolationMask')
                model.train()
            history.append(point); print(json.dumps(point),flush=True)
        if checkpoint_every>0 and step%checkpoint_every==0 and step>0:
            persist(step,time.perf_counter()-start,history)
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    persist(steps-1,time.perf_counter()-start,history)
    return out/'model.json'


def predict(atlas: Path, model_path: Path, device: str='cpu') -> tuple[np.ndarray, dict, float]:
    """Decode a saved checkpoint onto the atlas lattice. Returns heights, metadata and seconds.

    The checkpoint must have been trained on this exact reference array. The atlas content id
    also hashes the build code, so the reference hash is the identity that matters here.
    """
    import torch
    from safetensors.torch import load_file
    from geoneural.neural.models import make_model
    m=read_json(atlas); meta=read_json(model_path)
    if meta['reference_sha256']!=m['reference_sha256']: raise ValueError('Checkpoint belongs to another reference')
    # The checkpoint stores the lattice it was trained against while this reads the
    # atlas's copy. A mismatch would recompute every tile index and silently decode a
    # shuffled field, so compare them explicitly.
    if int(meta['page_intervals'])!=int(m['page_intervals']) or int(meta['reference_side'])!=int(np.load(atlas.parent/'reference.npy',mmap_mode='r',allow_pickle=False).shape[0]):
        raise ValueError('Checkpoint lattice disagrees with the atlas; tile indices would not match')
    weights=model_path.parent/'weights.safetensors'
    if sha_file(weights)!=meta['weights_sha256']: raise ValueError('Checkpoint weights changed')
    refpath=atlas.parent/'reference.npy'
    if sha_file(refpath)!=m['reference_sha256']: raise ValueError('Reference changed')
    reference=np.load(refpath,mmap_mode='r',allow_pickle=False)
    model=make_model(meta['model']).to(device)
    model.load_state_dict(load_file(str(weights))); model.eval()
    prediction=np.empty(reference.size,dtype=np.float32)
    start=time.perf_counter()
    with torch.inference_mode():
        for begin in range(0,reference.size,65536):
            indexes=np.arange(begin,min(begin+65536,reference.size))
            coords,tiles=features(indexes,reference.shape[0],m['page_intervals'],meta['model']['kind']=='shared')
            decoded=model(torch.from_numpy(coords).to(device),torch.from_numpy(tiles).to(device)).squeeze(-1)
            prediction[begin:begin+len(indexes)]=decoded.cpu().numpy()*meta['scale_m']+meta['mean_m']
    return prediction.reshape(reference.shape),meta,time.perf_counter()-start


def bake(atlas: Path, model_path: Path, out: Path,device: str='cpu') -> Path:
    from geoneural.data.build import pack
    m=read_json(atlas)
    prediction,meta,seconds=predict(atlas,model_path,device)
    reference=np.load(atlas.parent/'reference.npy',allow_pickle=False)
    start=time.perf_counter()-seconds
    errors=np.abs(prediction.astype(np.float64)-reference)
    fidelity={'mae_m':float(errors.mean()),'rmse_m':float(np.sqrt(np.mean(errors**2))),'p95_m':float(np.quantile(errors,.95)),'p99_m':float(np.quantile(errors,.99)),'max_m':float(errors.max()),'bake_seconds':time.perf_counter()-start,'samples':int(reference.size),'weights_bytes':meta['weights_bytes'],'model_metadata_bytes':model_path.stat().st_size}
    config=dict(m['config']); config['title']='NEURAL BAKED EXPERIMENT: '+config['title']
    provenance={'source':{'source_kind':'neural-baked-experiment','reference_atlas':m['content_id'],'reference_attribution':m.get('provenance',{}).get('source',{})},'model_sha256':sha_file(model_path),'weights_sha256':meta['weights_sha256'],'neural_error_against_observed_reference':fidelity,'warning':'The viewer reads conventional baked pages. Its load/decode timings do not measure neural inference. Codec benchmark errors on this atlas measure repacking predictions, not prediction fidelity.'}
    return pack(prediction,config,out,provenance)
