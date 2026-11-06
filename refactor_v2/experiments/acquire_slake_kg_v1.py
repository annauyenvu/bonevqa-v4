"""Acquire the author-linked, revision-pinned KG as an offline preparation artifact.

Only the KG archive and dataset card are fetched, never QA splits or images.
Archive entries are inventoried without extraction or unpickling.
"""
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];O=ROOT/'refactor_v2/reports/pf_slake_source_v1'
REV='a9083ce6c34ac3ffb17671a605962924d8a8f9e9'
EXPECTED='c7637f47772ce8ce028737aa3068496adc5fe774afb290719c5affd7c074af32'
def fetch(name):
    url=f'https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/{REV}/{name}'
    with urllib.request.urlopen(url,timeout=30) as response:
        data=response.read(1024*1024+1)
    assert len(data)<=1024*1024
    return data,url
def write(name,v):
    with (O/name).open('x') as f:json.dump(v,f,indent=2)
def main():
    assert not O.exists();O.mkdir()
    try:
        archive,url=fetch('KG.zip');digest=hashlib.sha256(archive).hexdigest();assert digest==EXPECTED
        card,card_url=fetch('README.md');assert b'license: cc-by-4.0' in card
        with (O/'KG.zip').open('xb') as f:f.write(archive)
        with (O/'SOURCE_README.md').open('xb') as f:f.write(card)
        entries=[]
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for info in z.infolist():
                assert not Path(info.filename).is_absolute() and '..' not in Path(info.filename).parts
                assert info.file_size<=20*1024*1024
                entry={'name':info.filename,'bytes':info.file_size,'compressed_bytes':info.compress_size,'directory':info.is_dir()}
                if not info.is_dir():entry['sha256']=hashlib.sha256(z.read(info)).hexdigest()
                entries.append(entry)
        receipt={'official_source_page':'https://www.med-vqa.com/slake/','dataset_card':card_url,'author_repository':'https://github.com/GaryGuTC/LaPA_model','revision':REV,'archive_url':url,'archive_sha256':digest,'archive_bytes':len(archive),'dataset_card_sha256':hashlib.sha256(card).hexdigest(),'declared_license':'CC-BY-4.0','entries':entries,'QA_splits_downloaded':False,'images_downloaded':False,'graph_transformation_or_training_released':False,'pickle_loaded':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        write('source_receipt.json',receipt);print(json.dumps({'verified_archive':True,'bytes':len(archive),'entries':entries}))
    except Exception as e:
        write('acquisition_failure.json',{'error':repr(e),'preserve_all_outputs':True});raise
if __name__=='__main__':main()
