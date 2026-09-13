#!/usr/bin/env python3
"""Register already-downloaded elevation GeoTIFFs with explicit source provenance.

Copies files into a fresh acquisition directory; originals remain untouched.
Vertical reference is an explicit human confirmation, not inferred from EPSG:25832.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from urllib.parse import urlparse
import rasterio
from geoneural.common import digest, region, sha_file, utc, write_json


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('files',type=Path,nargs='+')
    p.add_argument('--preset',default='essen-ruhr')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-url',required=True)
    p.add_argument('--provider',required=True)
    p.add_argument('--license',required=True)
    p.add_argument('--license-url',required=True)
    p.add_argument('--confirm-vertical-crs',choices=['EPSG:7837'],required=True)
    p.add_argument('--max-mib',type=int,default=2048)
    a=p.parse_args()
    if a.out.exists(): p.error('Choose a fresh output directory; no source or atlas is overwritten')
    if not 1<=len(a.files)<=256 or a.max_mib<=0: p.error('Require 1..256 bounded input files')
    for address in (a.source_url,a.license_url):
        if urlparse(address).scheme not in ('https','http'): p.error('Source and license URLs must be explicit HTTP(S) references')
    total=0
    for path in a.files:
        if not path.is_file(): p.error(f'Input is not a file: {path}')
        total+=path.stat().st_size
        with rasterio.open(path) as dataset:
            if dataset.count!=1 or dataset.crs is None:
                p.error(f'Require a single georeferenced elevation band: {path}')
    if total>a.max_mib*1024*1024: p.error('Input copy exceeds --max-mib')
    a.out.mkdir(parents=True)
    records=[]
    for index,path in enumerate(a.files):
        name=f'import-{index:03d}.tif'
        target=a.out/name
        shutil.copyfile(path,target)
        receipt={'path':name,'url':a.source_url,'sha256':sha_file(target),'bytes':target.stat().st_size,'registered_utc':utc(),'acquisition_time':'not asserted by importer','local_import':True}
        write_json(target.with_suffix('.tif.receipt.json'),receipt)
        records.append(receipt)
    config=region(a.preset)
    write_json(a.out/'input.json',{'schema':'geoneural-input-v1','config':config,'config_sha256':digest(config),'source_kind':'imported-observed-derived-dtm','provider':a.provider,'documentation':a.source_url,'license':a.license,'license_url':a.license_url,'vertical_crs':a.confirm_vertical_crs,'vertical_confirmation':'explicit importer assertion; no automatic vertical transform','files':records,'bytes':total})
    print(a.out/'input.json')


if __name__=='__main__': main()
