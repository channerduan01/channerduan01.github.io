# Git Tips




## Branch 

### List
git branch -a


### Create 
create a local branch from current branch:
git checkout -b <branch_name>

create remote branch by push some local branch:
git push origin <local_branch_name>:<remote_branch_name>


### Delete
for local branch:
git branch -D <branch_name>

for remote branch:
git push origin --delete <branch_name>


### Force drop local commit 
git reset --hard FETCH_HEAD




git clone http://gitlab-sc.alipay-inc.com/mad/mermaid


git clone git@gitlab.alipay-inc.com:mad/mermaid.git

cd mermaid/app/alipay/mkt/prod/huizhifu



