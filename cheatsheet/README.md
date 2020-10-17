# Dash でチートシートの作成
* https://qiita.com/akira-hamada/items/9e95ca60880f7fa6acf9


## SETUP
```bash
$ sudo gem install cheatset # dashが読み込むdocsetファイルの生成に必要
$ xcode-select --install # Xcode Xcode Command Line Toolsをインストールしていない場合は実行
```

## IMPLEMTENT
```bash
$ vim git_commands.rb # チートシートのファイル名(任意)。ファイルの内容は後述
$ cheatset generate git_commands.rb # docsetファイルに変換
$ open git_commands.docset # 生成されたdocsetをdashに読み込む
```


## CHEATSHEETs
* [R](https://rstudio.com/resources/cheatsheets/)
* [ggplot: ggplot2-cheatsheet-2.0-ja.pdf](https://rstudio.com/wp-content/uploads/2016/10/ggplot2-cheatsheet-2.0-ja.pdf)
* [ggplot: intro_ggplot2.html](https://kazutan.github.io/fukuokaR11/intro_ggplot2.html)
* [dplyr & tidyr](https://rstudio.com/wp-content/uploads/2015/09/data-wrangling-japanese.pdf)
