"""Preserve the completed Task 5.1/5.2 audit, papers and citation metadata."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/referee/task_5_1_5_2_20261012'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def run():
    audit=json.loads((OUT/'source_audit.json').read_text())
    for relative,expected in audit['sha256'].items():
        path=ROOT/relative
        assert digest(path)==expected,relative
        destination=OUT/'snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    records=[
        dict(key='WillemsVecchioKalogera2008',pdf='willems_prl_100_041102.pdf',
             url='https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.100.041102',
             doi='10.1103/PhysRevLett.100.041102',version='Published PRL article',
             verified_locations='pp. 041102-2 and 041102-4; Eqs. 1-3, Fig. 1, astrophysical implications',
             topic='Tidal, rotational and GR apsidal motion'),
        dict(key='WillemsVecchioKalogera2008Erratum',pdf='willems_erratum_prl_101_219903.pdf',
             url='https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.101.219903',
             doi='10.1103/PhysRevLett.101.219903',version='Published PRL erratum',
             verified_locations='Entire one-page erratum',topic='Correction to description of Seto frequency-resolution method'),
        dict(key='StroeerNelemans2009',pdf='stroeer_0909.1796v1.pdf',url='https://arxiv.org/pdf/0909.1796v1',
             doi='10.1111/j.1745-3933.2009.00754.x',version='arXiv 0909.1796v1, accepted manuscript',
             verified_locations='Sec. 2.1 Eqs. 1-8; Sec. 2.2 Eqs. 13-14; Sec. 4',
             topic='AM CVn evolution with mass transfer and tidal angular-momentum exchange',metadata='stroeer_metadata.json'),
        dict(key='WangCaoZhang2023',pdf='wang_2302.08814v3.pdf',url='https://arxiv.org/pdf/2302.08814v3',
             doi='10.1093/mnras/stad2358',version='arXiv 2302.08814v3',
             verified_locations='Sec. 2, Eqs. 14-25, circular conservative-transfer assumptions',
             topic='Mass transfer contribution to chirp',metadata='wang_metadata.json'),
        dict(key='XuanPengChen2021',pdf='xuan_2012.00049v1.pdf',url='https://arxiv.org/pdf/2012.00049v1',
             doi='10.1093/mnras/stab331',version='arXiv 2012.00049v1',
             verified_locations='Sec. 2 Eqs. 5-12; Sec. 3 confusion criteria; Sec. 7',
             topic='Source acceleration and circular-vacuum chirp-mass interpretation',metadata='xuan_metadata.json')]
    for record in records:
        path=OUT/'literature'/record['pdf']
        assert path.read_bytes().startswith(b'%PDF-')
        info=subprocess.run(['pdfinfo',str(path)],capture_output=True,text=True,check=True).stdout
        record['pages']=int(next(line.split(':')[1].strip() for line in info.splitlines() if line.startswith('Pages:')))
        record['pdf_sha256']=digest(path)
        record['full_relevant_sections_reviewed']=True
        if 'metadata' in record:
            record['metadata_source']='https://api.crossref.org/works/'+record['doi']
    write_json(OUT/'literature_evidence.json',dict(status='focused_primary_literature_verification_completed',
        reviewed_utc=datetime.now(timezone.utc).isoformat(),records=records,
        visually_checked_pages=['Willems published p2','Willems erratum p1','Stroeer p2','Wang p2','Xuan p3'],
        findings_report='reports/referee_task_5_1_5_2_audit.md',
        scope='Focused support for referee Comment 5; no quantitative omitted-effect forecast for the eGB grid'))
    entries=[
        ('WillemsVecchioKalogera2008',dict(author='Willems, B. and Vecchio, A. and Kalogera, V.',
            title='Probing White Dwarf Interiors with LISA: Periastron Precession in Eccentric Double White Dwarfs',
            journal='Physical Review Letters',volume='100',pages='041102',year='2008',
            doi='10.1103/PhysRevLett.100.041102',eprint='0706.3700',archivePrefix='arXiv')),
        ('WillemsVecchioKalogera2008Erratum',dict(author='Willems, B. and Vecchio, A. and Kalogera, V.',
            title='Erratum: Probing White Dwarf Interiors with LISA: Periastron Precession in Eccentric Double White Dwarfs',
            journal='Physical Review Letters',volume='101',pages='219903',year='2008',doi='10.1103/PhysRevLett.101.219903'))]
    for key,filename,eprint in [('StroeerNelemans2009','stroeer_metadata.json','0909.1796'),
        ('WangCaoZhang2023','wang_metadata.json','2302.08814'),('XuanPengChen2021','xuan_metadata.json','2012.00049')]:
        metadata=json.loads((OUT/'literature'/filename).read_text())['message']
        fields=dict(author=' and '.join(a['family']+', '+a['given'] for a in metadata['author']),
            title=metadata['title'][0],journal=metadata['container-title'][0],volume=metadata['volume'],
            number=metadata['issue'],pages=metadata['page'].replace('-','--'),
            year=str(metadata['published']['date-parts'][0][0]),doi=metadata['DOI'],
            eprint=eprint,archivePrefix='arXiv')
        entries.append((key,fields))
    bibliography='\n\n'.join('@article{'+key+',\n'+''.join('    '+k+' = {'+v+'},\n' for k,v in fields.items())+'}' for key,fields in entries)+'\n'
    (OUT/'reference_additions.bib').write_text(bibliography)
    write_json(OUT/'manuscript_claims.json',dict(source='reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex',
        source_sha256=audit['sha256']['reports/Effect_of_eccentricity_for_GB_in_LISA-final/apssamp.tex'],
        audit_locations=[108,146,300,396,582,743,935,1030,1246],
        findings_report='reports/referee_task_5_1_5_2_audit.md',edits_applied=False,
        frequency_label_issue='Grid constructor uses mean_motion=pi*f0, whereas text calls f0 orbital frequency',
        fdot_interface_issue='NumPy source ignores fdot; fixed JAX source implements it; non-fixed links reject it'))
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json']
    paths.extend([ROOT/'reports/referee_task_5_1_5_2_audit.md',Path(__file__).resolve()])
    write_json(OUT/'manifest.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        task_5_1_status='completed_source_contract_and_manuscript_claim_audit',
        task_5_2_status='completed_focused_primary_literature_verification',
        source_implementation_modified=False,manuscript_edits_applied=False,
        sha256={str(p.relative_to(ROOT)):digest(p) for p in paths},
        inspected_source_hashes=audit['sha256'],
        next_task='5.3 applicability assessment',
        limitations=['No physical omitted-effect amplitude or global validity boundary calculated',
            'Existing fdot interface asymmetry documented without changing the frozen implementation']))
    print(json.dumps(dict(task_5_1='complete',task_5_2='complete',primary_documents=len(records),
        prepared_bibliography_entries=len(entries),preserved_files=len(paths)),indent=2))


if __name__=='__main__':
    run()
