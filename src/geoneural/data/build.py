"""Prepare a canonical regional raster, then publish independent coarse pages.

The reference grid is regional and bounded. This is not a planet-sized in-memory
preprocessor. Filtering occurs before splitting pages, so shared level boundaries
come from identical samples. Renderer skirts can hide cross-LOD seams; they do not
certify them.
"""
from __future__ import annotations
import hashlib
import math
import os
import shutil
import uuid
from pathlib import Path
import numpy as np
from geoneural.codecs import eat1 as codec
from geoneural.common import digest, environment, read_json, sha_file, utc, write_json

#: Reprojected tile nodes within this many samples of a tile edge lack full source
#: support and are discarded in favour of a neighbouring tile's interior nodes.
DEFAULT_EDGE_TRIM = 2


def grid_shape(config: dict, max_mib: int = 512) -> int:
    west,south,east,north=map(float,config['bbox'])
    spacing=float(config['spacing_m'])
    intervals=int(config['page_intervals'])
    if not all(map(math.isfinite,(west,south,east,north,spacing))) or spacing<=0 or east<=west or north<=south:
        raise ValueError('Invalid grid bounds/spacing')
    n=round((east-west)/spacing)
    if abs(n*spacing-(east-west))>1e-5 or abs((north-south)-(east-west))>1e-5:
        raise ValueError('The grid must be a square with an integral interval count')
    if not 8<=intervals<=128 or intervals & (intervals-1):
        raise ValueError('Page intervals must be a power of two in 8..128')
    pages=n//intervals
    if n%intervals or pages<1 or pages & (pages-1) or n>8192:
        raise ValueError('The grid must contain a dyadic number of pages, with at most 8192 intervals')
    if (n+1)**2*48 > max_mib*1024*1024:
        raise ValueError('Conservative regional preprocessing workspace exceeds --max-mib; split the region')
    return n+1


def smooth_decimate(grid: np.ndarray) -> np.ndarray:
    result=np.asarray(grid,dtype=np.float64)
    for axis in (0,1):
        padding=[(0,0),(0,0)]
        padding[axis]=(2,2)
        padded=np.pad(result,padding,mode='edge')
        slices=[]
        for k,weight in enumerate((1,4,6,4,1)):
            selection=[slice(None),slice(None)]
            selection[axis]=slice(k,k+result.shape[axis])
            slices.append(padded[tuple(selection)]*(weight/16.0))
        result=sum(slices)
    return result[::2,::2].astype(np.float32)


def expand_bilinear(grid: np.ndarray, factor: int) -> np.ndarray:
    if factor==1:
        return grid.astype(np.float64)
    side=grid.shape[0]
    samples=np.arange((side-1)*factor+1,dtype=np.float64)/factor
    positions=np.arange(side)
    horizontal=np.stack([np.interp(samples,positions,row) for row in grid])
    return np.stack([np.interp(samples,positions,col) for col in horizontal.T],axis=1)


def erode_valid(mask: np.ndarray, samples: int) -> np.ndarray:
    """Shrink a valid-data mask inward by `samples` nodes.

    Each downloaded tile is requested with a halo so that reprojection has source
    support on every side. The outermost reprojected nodes still lack full
    support, so bilinear warping biases them. The halo exists to be discarded;
    keeping it is what makes two tiles disagree about the same ground.
    """
    if samples <= 0:
        return mask
    out = mask.copy()
    for _ in range(samples):
        shrunk = out.copy()
        shrunk[1:, :] &= out[:-1, :]
        shrunk[:-1, :] &= out[1:, :]
        shrunk[:, 1:] &= out[:, :-1]
        shrunk[:, :-1] &= out[:, 1:]
        out = shrunk
    return out

def nodata_report(merged: np.ndarray, config: dict, out: Path) -> Path:
    """Record where a refused lattice is missing data.

    Refusing to publish an incomplete atlas is the right policy, but a bare count
    gives an operator nothing to act on. This writes the mask and the projected
    bounding box of the missing samples so the gap can be identified as a service
    boundary, a state border or a genuine hole in the source.
    """
    west,south,east,north=map(float,config['bbox'])
    spacing=float(config['spacing_m'])
    holes=~np.isfinite(merged)
    rows,cols=np.nonzero(holes)
    extent=None
    if rows.size:
        extent={'row_range':[int(rows.min()),int(rows.max())],'col_range':[int(cols.min()),int(cols.max())],
                'easting_range_m':[west+float(cols.min())*spacing,west+float(cols.max())*spacing],
                'northing_range_m':[north-float(rows.max())*spacing,north-float(rows.min())*spacing]}
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'nodata-mask.npy',holes,allow_pickle=False)
    report={'schema':'geoneural-nodata-report-v1','created_utc':utc(),'config':config,
            'sample_side':int(merged.shape[0]),'missing_samples':int(rows.size),
            'missing_fraction':float(rows.size)/float(merged.size),'extent':extent,
            'mask':'nodata-mask.npy','environment':environment(),
            'note':'Publication was refused. Missing data is never zero height and never certified empty terrain.'}
    write_json(out/'nodata-report.json',report)
    return out/'nodata-report.json'

def prepare(input_path: Path, out: Path, max_mib: int = 512, edge_trim: int = DEFAULT_EDGE_TRIM) -> Path:
    merged,config,overlap_differences,observed,transform=merge(input_path,max_mib,edge_trim)
    missing=int(np.count_nonzero(~np.isfinite(merged)))
    if missing:
        report=nodata_report(merged,config,input_path.parent)
        raise ValueError(
            f'{missing} canonical samples are nodata ({100.0*missing/merged.size:.3f}% of the lattice) '
            f'at edge_trim={edge_trim}. Zero-filling is forbidden. The hole locations are recorded at '
            f'{report}. Either acquire the missing coverage, or lower --edge-trim and accept the '
            'reprojection edge bias it reintroduces; that trade-off is deliberate, not automatic.')
    overlap=np.concatenate(overlap_differences) if overlap_differences else np.zeros(1)
    overlap_statistics={'samples':int(overlap.size),'max_m':float(overlap.max()),
        'p50_m':float(np.percentile(overlap,50)),'p99_m':float(np.percentile(overlap,99)),
        'note':'Disagreement between tiles about the same ground after edge trimming. '
               'A nonzero value means the prepared reference depends on tile decomposition.'}
    provenance={'input_sha256':sha_file(input_path),'source':read_json(input_path),'source_rasters':observed,'resampling':f'rasterio bilinear; deterministic sorted first-valid overlap owner; halo edge trimmed by {edge_trim} samples','edge_trim_samples':edge_trim,'overlap_max_difference_m':overlap_statistics['max_m'],'overlap_statistics':overlap_statistics,'transform':list(transform)[:6]}
    return pack(merged,config,out,provenance,max_mib)


def merge(input_path: Path, max_mib: int = 512, edge_trim: int = DEFAULT_EDGE_TRIM):
    """The canonical lattice from the acquired tiles, NaN where no tile covers a node. Returns the lattice, the
    config, the overlap differences, the source raster descriptions and the transform."""
    import rasterio
    from rasterio.transform import Affine
    from rasterio.warp import reproject, Resampling
    source=read_json(input_path)
    if source.get('schema')!='geoneural-input-v1':
        raise ValueError('Expected a complete terrain input.json')
    config=source['config']
    n=grid_shape(config,max_mib)
    west,south,east,north=map(float,config['bbox'])
    spacing=float(config['spacing_m'])
    transform=Affine(spacing,0,west-spacing/2,0,-spacing,north+spacing/2)
    if not 0<=edge_trim<=16:
        raise ValueError('edge_trim must be 0..16 samples')
    merged=np.full((n,n),np.nan,dtype=np.float32)
    overlap_differences=[]
    observed=[]
    files=source.get('files',[])
    if not files or len(files)>256:
        raise ValueError('Input must contain 1..256 GeoTIFFs')
    for item in sorted(files,key=lambda x:x['path']):
        path=(input_path.parent/item['path']).resolve()
        if not path.is_relative_to(input_path.parent.resolve()):
            raise ValueError('Input file escapes the acquisition directory')
        if sha_file(path)!=item['sha256']:
            raise ValueError(f'Source changed after acquisition: {path}')
        temporary=np.full_like(merged,np.nan)
        with rasterio.open(path) as dataset:
            if dataset.crs is None or dataset.count!=1:
                raise ValueError(f'Expected one georeferenced elevation band: {path}')
            observed.append({'path':item['path'],'crs':dataset.crs.to_string(),'transform':list(dataset.transform)[:6],'width':dataset.width,'height':dataset.height,'nodata':str(dataset.nodata)})
            reproject(source=rasterio.band(dataset,1),destination=temporary,src_transform=dataset.transform,src_crs=dataset.crs,src_nodata=dataset.nodata,dst_transform=transform,dst_crs=config['crs'],dst_nodata=np.nan,resampling=Resampling.bilinear,num_threads=1,warp_mem_limit=64)
        valid=erode_valid(np.isfinite(temporary),edge_trim)
        overlap=valid & np.isfinite(merged)
        if overlap.any():
            overlap_differences.append(np.abs(temporary[overlap]-merged[overlap]).astype(np.float64))
        take=valid & ~np.isfinite(merged)
        merged[take]=temporary[take]
    return merged,config,overlap_differences,observed,transform


def pack(reference: np.ndarray, config: dict, out: Path, provenance: dict, max_mib: int = 512) -> Path:
    import rasterio
    from rasterio.transform import Affine
    n=grid_shape(config,max_mib)
    if reference.shape!=(n,n) or not np.isfinite(reference).all():
        raise ValueError('Reference does not match the finite canonical grid')
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an atlas; choose a fresh --out: {out}')
    out.parent.mkdir(parents=True,exist_ok=True)
    stage=out.with_name(out.name+'.building-'+uuid.uuid4().hex)
    stage.mkdir()
    try:
        intervals=int(config['page_intervals'])
        spacing=float(config['spacing_m'])
        quantum=float(config['quantum_m'])
        levels=int(math.log2((n-1)//intervals))
        west,south,east,north=map(float,config['bbox'])
        reference=np.asarray(reference,dtype=np.float32)
        np.save(stage/'reference.npy',reference,allow_pickle=False)
        transform=Affine(spacing,0,west-spacing/2,0,-spacing,north+spacing/2)
        with rasterio.open(stage/'reference.tif','w',driver='GTiff',width=n,height=n,count=1,dtype='float32',crs=config['crs'],transform=transform,tiled=True,blockxsize=256,blockysize=256,compress='DEFLATE',predictor=3) as dst:
            dst.write(reference,1)
        pages={}
        grid=reference
        for level in range(levels+1):
            count=(grid.shape[0]-1)//intervals
            factor=1<<level
            width=intervals*spacing*factor
            for y in range(count):
                for x in range(count):
                    patch=grid[y*intervals:y*intervals+intervals+1,x*intervals:x*intervals+intervals+1]
                    packed=codec.encode(patch,level,quantum)
                    key=f'{level}/{x}/{y}'
                    path=stage/'pages'/str(level)/f'{x}_{y}.eat.gz'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(packed)
                    begin_y,begin_x=y*intervals*factor,x*intervals*factor
                    fine=reference[begin_y:begin_y+intervals*factor+1,begin_x:begin_x+intervals*factor+1]
                    error=float(np.max(np.abs(expand_bilinear(patch,factor)-fine)))+quantum/2
                    pages[key]={'path':path.relative_to(stage).as_posix(),'level':level,'x':x,'y':y,'side':intervals+1,'x_m':x*width,'z_m':y*width,'width_m':width,'spacing_m':spacing*factor,'packed_bytes':len(packed),'raw_bytes':32+patch.size*4,'sha256':hashlib.sha256(packed).hexdigest(),'min_m':float(fine.min())-quantum/2,'max_m':float(fine.max())+quantum/2,'sample_error_m':error,'error_kind':'sampled-finest-grid-bilinear-relative; not continuous-ground or triangle proof'}
            if level<levels:
                grid=smooth_decimate(grid)
        algorithm=digest({p.name:sha_file(p) for p in (Path(__file__),Path(codec.__file__))})
        manifest={'schema':'geoneural-atlas-v1','name':config['title'],'source_kind':provenance.get('source',{}).get('source_kind','derived-experiment'),'config':config,'crs':config['crs'],'vertical_crs':config['vertical_crs'],'bounds':config['bbox'],'sample_centres':True,'north_up':True,'sample_side':n,'spacing_m':spacing,'page_intervals':intervals,'max_level':levels,'root':f'{levels}/0/0','pages':pages,'height_range_m':[float(reference.min()),float(reference.max())],'runtime_page_bytes':sum(p['packed_bytes'] for p in pages.values()),'reference_raw_bytes':int(reference.nbytes),'reference_sha256':sha_file(stage/'reference.npy'),'algorithm_sha256':algorithm,'filter':'binomial-5 separable, edge-pad, decimate by two','quantum_m':quantum,'provenance':provenance,'environment':environment(),'created_utc':utc(),'qualification':'authored experimental pipeline; validate actual data and rendered output locally'}
        manifest['content_id']=digest(manifest)
        write_json(stage/'atlas.json',manifest)
        write_json(stage/'ATTRIBUTION.json',provenance.get('source',{}))
        os.rename(stage,out)
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print(f'Atlas: {out}/atlas.json; {len(pages)} pages, {manifest["runtime_page_bytes"]:,} compressed page bytes')
    return out/'atlas.json'
