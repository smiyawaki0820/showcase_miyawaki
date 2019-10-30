### github command
```code

# branch
$ git branch
$ git branch --delete [削除branch]
$ git checkout [移動先branch]
$ git checkout -b [作成branch]
$ git checkout -b [作成ローカルbranch] origin/[参照先リモートbranch]

# tag
$ git tag [tag]
$ git tag -d [tag]

# merge
$ git merge --no-ff [指定branch]   # 現在作業中branch に 指定branch を merge (merge後も戻せるようにする)

# push
$ git push origin [ローカルbranch]
```
