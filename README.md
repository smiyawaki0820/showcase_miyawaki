### github command
<a href="https://qiita.com/Iyutaka/items/248ebc0a0cc4ba8cb911v">Gitで開発ブランチにmasterの内容を反映させる方法 (git rebase)</a>
```code

# branch
$ git branch
$ git branch --delete [削除branch]
$ git checkout [移動先branch]
$ git checkout -b [作成branch]
$ git checkout -b [作成ローカルbranch] origin/[参照先リモートbranch]  # pull remote branch
$ git checkout [参照先branch] -- [file] # pull file from branch
git push --set-upstream origin [ブランチ]

# tag
$ git tag [tag]
$ git tag -d [tag]

# merge
$ git checkout [反映先branch] && git merge master  # master 内容を 反映先branch に反映
$ git merge --no-ff [指定branch]   # 現在作業中branch に 指定branch を merge (merge後も戻せるようにする)

# push
$ git push origin [ローカルbranch]

# delete
$ git rm --cached [flie]  # remote のみ削除
```

### vim
```code

# 置換 g:all, c:確認しながら
:%s/[置換前word]/[置換後word]/g
ctrl + p  #補間
```
