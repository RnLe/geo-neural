"""Opt-in local codec/IO measurements; never invoked implicitly by a viewer."""
from __future__ import annotations
import gzip
import hashlib
import random
import time
from pathlib import Path
import numpy as np
from geoneural.codecs.eat1 import decode, encode
from geoneural.common import environment, read_json, safe_child, sha_file, utc, write_json


def percentiles(values: list[float] | np.ndarray) -> dict:
    a=np.asarray(values,dtype=np.float64)
    return {k:float(v) for k,v in zip(('p50','p95','p99','max'),[*np.quantile(a,[.5,.95,.99]),np.max(a)])}


def run(atlas_path: Path, out: Path, repeats: int = 3, limit: int = 128, seed: int = 1729) -> Path:
    if not 1<=repeats<=100 or not 1<=limit<=4096:
        raise ValueError('Invalid benchmark bounds')
    m=read_json(atlas_path)
    if m.get('schema')!='geoneural-atlas-v1':
        raise ValueError('Unsupported atlas')
    reference_path=atlas_path.parent/'reference.npy'
    if sha_file(reference_path)!=m['reference_sha256']:
        raise ValueError('Reference identity changed')
    reference=np.load(reference_path,mmap_mode='r',allow_pickle=False)
    keys=list(m['pages'])
    random.Random(seed).shuffle(keys)
    keys=keys[:limit]
    observations=[]
    for repeat in range(repeats):
        for key in keys:
            meta=m['pages'][key]
            path=safe_child(atlas_path.parent,meta['path'])
            start=time.perf_counter_ns()
            payload=path.read_bytes()
            loaded=time.perf_counter_ns()
            # SHA is measured independently: distinguish integrity cost from decoding.
            verified=hashlib.sha256(payload).hexdigest()==meta['sha256']
            hashed=time.perf_counter_ns()
            if not verified:
                raise ValueError(f'Corrupt page {key}')
            heights,header=decode(payload,meta['raw_bytes'])
            end=time.perf_counter_ns()
            observations.append({'repeat':repeat,'key':key,'level':meta['level'],'samples':int(heights.size),'read_ms':(loaded-start)/1e6,'sha_ms':(hashed-loaded)/1e6,'decode_ms':(end-hashed)/1e6,'total_ms':(end-start)/1e6,'bytes':len(payload)})
    errors=[]
    for key in sorted(k for k,p in m['pages'].items() if p['level']==0)[:limit]:
        p=m['pages'][key]
        h,_=decode(safe_child(atlas_path.parent,p['path']).read_bytes(),p['raw_bytes'])
        i=int(m['page_intervals']); x,y=p['x']*i,p['y']*i
        errors.append(np.abs(h-reference[y:y+i+1,x:x+i+1]).ravel())
    codec_results=[]
    leaf=next(p for p in m['pages'].values() if p['level']==0)
    side=leaf['side']; i=m['page_intervals']; x,y=leaf['x']*i,leaf['y']*i
    patch=np.array(reference[y:y+side,x:x+side],dtype=np.float32)
    for name,encode_fn,decode_fn in [
        ('float32-gzip',lambda:gzip.compress(patch.astype('<f4').tobytes(),mtime=0),lambda b:np.frombuffer(gzip.decompress(b),dtype='<f4').reshape(patch.shape)),
        ('EAT1-q32-gzip',lambda:encode(patch,0,m['quantum_m']),lambda b:decode(b)[0])]:
        start=time.perf_counter_ns(); payload=encode_fn(); middle=time.perf_counter_ns(); restored=decode_fn(payload); end=time.perf_counter_ns()
        codec_results.append({'codec':name,'scope':'one illustrative leaf, not a full codec comparison','bytes':len(payload),'bits_per_sample':len(payload)*8/patch.size,'encode_ms':(middle-start)/1e6,'decode_ms':(end-middle)/1e6,'max_error_m':float(np.abs(restored-patch).max())})
    complete_bytes=m['runtime_page_bytes']+atlas_path.stat().st_size+(atlas_path.parent/'ATTRIBUTION.json').stat().st_size
    report={'schema':'geoneural-benchmark-v1','created_utc':utc(),'atlas_content_id':m['content_id'],'environment':environment(),'repeats':repeats,'page_limit':limit,'seed':seed,'cache_semantics':'first application read then repeated reads; OS cache not flushed, no process-cold or disk-cold claim','runtime_archive_bytes':complete_bytes,'reference_raw_bytes':m['reference_raw_bytes'],'ratio_reference_to_all_runtime_bytes':m['reference_raw_bytes']/complete_bytes,'summary_ms':{stage:percentiles([o[stage] for o in observations]) for stage in ('read_ms','sha_ms','decode_ms','total_ms')},'level_summary_ms':{str(level):percentiles([o['total_ms'] for o in observations if o['level']==level]) for level in sorted({o['level'] for o in observations})},'leaf_quantization_error_m':percentiles(np.concatenate(errors)),'error_sampling':'bounded sorted leaf subset; shared border nodes duplicated; sampled coverage explicitly limited','codec_pilot':codec_results,'observations':observations,'qualification':'local source/IO diagnostic only; not whole-renderer performance, statistical significance or source accuracy'}
    write_json(out,report)
    return out
