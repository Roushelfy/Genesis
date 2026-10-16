import argparse,datetime,hashlib,json,os,subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--label',required=True)
parser.add_argument('--archive',action='append',default=[])
parser.add_argument('--sources',action='store_true')
parser.add_argument('--file-list',type=Path)
parser.add_argument('--message',required=True)
a=parser.parse_args()
wt=Path('/mnt/home/zhaofeng/workspace/Genesis/.worktrees/rigid-stress-recovery')
main=Path('/mnt/home/zhaofeng/workspace/Genesis')
p=Path('/mnt/data/zhaofeng/projects/workspace/Genesis/rigid-stress-recovery/runs/20261014-physical-validation')
assert a.message.endswith('.')
def git(*args,cwd=wt):return subprocess.check_output(['git',*args],cwd=cwd)
assert git('branch','--show-current').decode().strip()=='rigid-stress-recovery'
assert git('rev-parse','HEAD',cwd=main).decode().strip()=='53da6e92c8c7c804413f6ca1378bb5b8577d6c11'
assert git('remote','get-url','fork').decode().strip()=='https://github.com/Roushelfy/Genesis'
assert not git('diff','--cached','--name-only','-z')
base=git('rev-parse','HEAD').decode().strip()
remote_before=git('ls-remote','fork','refs/heads/rigid-stress-recovery').decode().split()[0]
assert remote_before==base,(remote_before,base)
paths=set()
if a.sources:
 paths.update(git('diff','--name-only','-z').decode().rstrip('\0').split('\0'))
 paths.update(git('ls-files','--others','--exclude-standard','-z','--',':(glob)research/rigid_stress/*.py',':(glob)research/rigid_stress/*.patch',':(glob)research/rigid_stress/scripts/*',':(glob)research/rigid_stress/docs/*').decode().rstrip('\0').split('\0'))
archive_files=set()
for label in a.archive:
 root=wt/'research/rigid_stress/evidence/20261014-physical'/label
 assert (root/'manifest.json').exists()
 archive_files.update(str(f.relative_to(wt)) for f in root.rglob('*') if f.is_file())
if a.file_list:
 selected=set(json.loads(a.file_list.read_text()))
 assert selected and selected <= archive_files
 paths.update(selected)
else:
 paths.update(archive_files)
paths.discard('');paths=sorted(paths)
assert paths
rows=[]
for relative in paths:
 f=wt/relative;size=f.stat().st_size
 sha256=hashlib.sha256();sha1=hashlib.sha1(('blob '+str(size)+'\0').encode())
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):sha256.update(block);sha1.update(block)
 rows.append({'path':relative,'bytes':size,'sha256':sha256.hexdigest(),'git_blob':sha1.hexdigest()})
assert sum(r['bytes'] for r in rows)<1_700_000_000
proof=p/'meta'/f'{a.label}-git-inventory-v1.json';assert not proof.exists()
record={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'branch':'rigid-stress-recovery','base_revision':base,'main_revision_preserved':'53da6e92c8c7c804413f6ca1378bb5b8577d6c11','files':rows,'total_bytes':sum(r['bytes'] for r in rows),'archives':a.archive,'sources_included':a.sources,'message':a.message,'archive_file_subset':str(a.file_list) if a.file_list else None,'archive_file_subset_sha256':hashlib.sha256(a.file_list.read_bytes()).hexdigest() if a.file_list else None}
proof.write_text(json.dumps(record,indent=2)+'\n')
for start in range(0,len(paths),200):subprocess.run(['git','add','-f','--',*paths[start:start+200]],cwd=wt,check=True)
staged=set(git('diff','--cached','--name-only','-z').decode().rstrip('\0').split('\0'))
assert staged==set(paths),(staged-set(paths),set(paths)-staged)
entries=git('ls-files','--stage','-z','--',*paths).split(b'\0');actual={}
for entry in entries:
 if entry:
  data,name=entry.split(b'\t',1);mode,oid,stage=data.decode().split();assert stage=='0';actual[name.decode()]=oid
assert actual=={r['path']:r['git_blob'] for r in rows}
subprocess.run(['git','diff','--cached','--check','--','.',':(exclude)research/rigid_stress/evidence/**'],cwd=wt,check=True)
print('Verified staging',a.label,len(rows),record['total_bytes'],flush=True)
tmp=p/'meta/git-tmp'/a.label;tmp.mkdir(parents=True,exist_ok=False)
with (p/'meta'/f'{a.label}-commit-v1.log').open('w') as out:
 result=subprocess.run(['git','commit','-m',a.message],cwd=wt,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'TMPDIR':str(tmp)})
assert result.returncode==0,result.returncode
head=git('rev-parse','HEAD').decode().strip()
assert set(git('diff-tree','--no-commit-id','--name-only','-r','-z',head).decode().rstrip('\0').split('\0'))==set(paths)
tree=git('ls-tree','-r','-z',head,'--',*paths).split(b'\0');tree_oids={}
for entry in tree:
 if entry:
  data,name=entry.split(b'\t',1);mode,kind,oid=data.decode().split();assert kind=='blob';tree_oids[name.decode()]=oid
assert tree_oids==actual
assert git('rev-parse','HEAD',cwd=main).decode().strip()==record['main_revision_preserved']
local={'revision':head,'exact_changed_paths':True,'all_index_and_tree_blob_ids_match_actual_bytes':True,'main_unchanged':True,'file_count':len(rows),'raw_bytes':record['total_bytes']}
(p/'meta'/f'{a.label}-local-verification-v1.json').write_text(json.dumps(local,indent=2)+'\n')
print('Committed',head,flush=True)
with (p/'meta'/f'{a.label}-push-v1.log').open('w') as out:
 result=subprocess.run(['git','push','fork','HEAD:refs/heads/rigid-stress-recovery'],cwd=wt,stdout=out,stderr=subprocess.STDOUT,env={**os.environ,'TMPDIR':str(tmp)})
assert result.returncode==0,result.returncode
remote=git('ls-remote','fork','refs/heads/rigid-stress-recovery').decode().split()[0]
assert remote==head
assert git('rev-parse','HEAD',cwd=main).decode().strip()==record['main_revision_preserved']
(p/'meta'/f'{a.label}-remote-verification-v1.json').write_text(json.dumps({**local,'remote':'fork','remote_revision':remote,'remote_head_matches':True},indent=2)+'\n')
print('Pushed and verified',a.label,head,flush=True)
