#!/usr/bin/env python3
"""Inspect acquired geological GML; optionally rasterize a reviewed category field.

Fiona/GDAL can expose only part of a complex INSPIRE schema. This script never
selects a lithological attribute automatically. Referenced/nested properties may
need a separate curated parser. Rasterized classes are map interpretation, not
exact subsurface blocks. Category zero means UNKNOWN, not air or zero geology.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from geoneural.common import environment, read_json, safe_child, sha_file, write_json
from geoneural.data.gml import inventory as gml_inventory


def inventory(manifest: Path) -> dict:
    """Inventory the retained GML schema.

    This reads the GML directly rather than through a compiled vector driver. The
    schema question (which feature types, fields and linked references exist)
    is answerable from the document, and doing it this way keeps the inventory
    available on interpreters where no Fiona wheel is published. Verified source
    hashes are still required: an inventory of changed input is not an inventory
    of the acquisition.
    """
    m = read_json(manifest)
    if m.get('schema') != 'geoneural-geology-v1':
        raise ValueError('Require a geology.json acquisition manifest')
    if not m.get('complete'):
        raise ValueError('Acquisition is incomplete; a partial catalogue is not a coverage statement')
    for item in m['files']:
        path = safe_child(manifest.parent, item['path'])
        if sha_file(path) != item['sha256']:
            raise ValueError(f'Geological input changed after acquisition: {path}')
    return gml_inventory(manifest)


def _vector_stack():
    """Import the rasterization stack only when a field is actually produced."""
    try:
        import fiona
        from rasterio.features import rasterize
        from rasterio.transform import Affine
        from rasterio.warp import transform_geom
    except ImportError as exc:
        raise ImportError(
            f'Producing a category raster needs Fiona and Rasterio ({exc}). The schema inventory '
            'does not. Install a Fiona build for this interpreter, or run the inventory only.'
        ) from exc
    return fiona, rasterize, Affine, transform_geom


def make_field(manifest: Path, atlas: Path, out: Path, layer: str, field: str, max_features: int) -> Path:
    fiona, rasterize, Affine, transform_geom = _vector_stack()
    m=read_json(manifest); a=read_json(atlas)
    if not m.get('complete') or m.get('schema')!='geoneural-geology-v1': raise ValueError('Incomplete geological input')
    if a.get('schema')!='geoneural-atlas-v1': raise ValueError('Unsupported atlas')
    if out.exists(): raise FileExistsError('Use a fresh field directory')
    if not 1<=max_features<=100000: raise ValueError('Invalid feature allowance')
    side=int(a['sample_side'])
    if side>4097: raise ValueError('This regional rasterizer is limited to 4097² nodes')
    shapes=[]; seen=set(); scanned=0; missing=0; duplicate=0
    for item in m['files']:
        path=safe_child(manifest.parent,item['path'])
        if sha_file(path)!=item['sha256']: raise ValueError('Geological source changed')
        if layer not in fiona.listlayers(path): continue
        with fiona.open(path,layer=layer) as src:
            if field not in src.schema['properties']: raise ValueError(f'{field} absent from {layer}: {src.schema}')
            if not src.crs: raise ValueError('Unknown geological geometry CRS')
            for feature in src:
                scanned+=1
                if scanned>max_features: raise ValueError('Feature cap reached; refuse incomplete field')
                value=feature['properties'].get(field)
                if feature['geometry'] is None or value is None:
                    missing+=1; continue
                category=str(value)
                geometry=transform_geom(src.crs,a['crs'],fiona.model.to_dict(feature['geometry']))
                # Stable content dedup is independent of page-local Fiona feature IDs.
                signature=json.dumps([geometry,category],sort_keys=True,separators=(',',':'))
                if signature in seen:
                    duplicate+=1; continue
                seen.add(signature)
                shapes.append((geometry,category))
    if not shapes: raise ValueError('No usable geometry/categories; inspect the schema and selected layer')
    categories=sorted({category for _,category in shapes})
    if len(categories)>65534: raise ValueError('Category vocabulary exceeds uint16')
    codebook={name:index+1 for index,name in enumerate(categories)}
    step=float(a['spacing_m']); west,_,_,north=a['bounds']
    transform=Affine(step,0,float(west)-step/2,0,-step,float(north)+step/2)
    values=rasterize([(g,codebook[c]) for g,c in shapes],out_shape=(side,side),transform=transform,fill=0,dtype='uint16',all_touched=False)
    out.mkdir(parents=True)
    np.save(out/'categories.npy',values,allow_pickle=False)
    meta={'schema':'geoneural-geology-categories-v1','source_geology_sha256':sha_file(manifest),'atlas_content_id':a['content_id'],'layer':layer,'field':field,'unknown_code':0,'codebook':codebook,'raster_sha256':sha_file(out/'categories.npy'),'raster_bytes':(out/'categories.npy').stat().st_size,'unknown_fraction':float(np.mean(values==0)),'scanned':scanned,'missing_geometry_or_category':missing,'duplicate_content':duplicate,'unique_features':len(shapes),'overlap_policy':'stable input order, later feature wins; inspect overlapping maps before model use','interpretation':'reviewed map attribute rasterized at atlas nodes; fine raster spacing adds no geological measurement resolution','attribution':m['attribution'],'environment':environment()}
    write_json(out/'context.json',meta)
    return out/'context.json'


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geology',type=Path,required=True)
    p.add_argument('--atlas',type=Path)
    p.add_argument('--layer'); p.add_argument('--field'); p.add_argument('--out',type=Path)
    p.add_argument('--max-features',type=int,default=10000)
    a=p.parse_args()
    if a.field is None:
        print(json.dumps(inventory(a.geology),indent=2,default=str))
    else:
        if not a.layer or a.atlas is None or a.out is None: p.error('--field requires explicit --layer, --atlas and fresh --out')
        print(make_field(a.geology,a.atlas,a.out,a.layer,a.field,a.max_features))


if __name__=='__main__': main()
