
# pre-process
コマンドラインで以下を入力

```setup
$ sudo gem install cheatset # dashが読み込むdocsetファイルの生成に必要
$ xcode-select --install # Xcode Xcode Command Line Toolsをインストールしていない場合は実行
```
```implement
$ vim git_commands.rb # チートシートのファイル名(任意)。ファイルの内容は後述
$ cheatset generate git_commands.rb # docsetファイルに変換
# cheatset command not foundのようなことを言われる場合は source .zshrcなりターミナルを閉じるなりしてからコマンド再実行
$ open git_commands.docset # 生成されたdocsetをdashに読み込む
```


# 参考
https://qiita.com/akira-hamada/items/9e95ca60880f7fa6acf9
