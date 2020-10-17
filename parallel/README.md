# ShellScript Tutorial
- http://motw.mods.jp/shellscript/tutorial.html
- https://shellscript.sunone.me/tutorial.html

## EXAMPLES
```bash
#!/usr/bin/bash
set -ex
USAGE="bash $0 [-g GPU] [-i INPUT]"

while getopts g:i: opt ; do
    case ${opt} in
        g ) FLG_G="TRUE"; gpu=${OPTARG};;
        i ) FLG_G="TRUE"; fi=${OPTARG};;
        * ) echo ${USAGE} 1>&2; exit 1 ;;
    esac
done

test "${FLG_G}" != "TRUE" && gpu=0
test "${FLG_I}" != "TRUE" && (echo $USAGE 1>&2; exit 1)

# RUN =========================
CUDA_VISIBLE_DEVICE=$gpu python  \
  --fi $fi
| tee $f_log
```

## シェルスクリプト 一行目
### shebang
```bash
#!/bin/bash -ev
```
> スクリプトを実行するためのインタプリタ（/bin/bash によってシェルスクリプトが解釈・実行される）
* `-e` : 実行コマンドが ~0 で終了した場合，即座に終了
* `-x` : 実行されたコマンドを表示（変数は展開される）
* `-v` : これから実行されるコマンドを表示（変数名を表示）

### 特別な変数
<img src="https://i.gyazo.com/82fbc3195e14527743c753d83e426feb.png" alt="sh special variables" title="http://motw.mods.jp/shellscript/tutorial.html">

* `c1 ; c2`     : 同一行に複数コマンド
* `c1 &`        : バックグラウンド実行
* `c1 && c2`    : c1 が正常実行されたら c2 を実行
* `c1 | c2`     : c1 の標準出力を c2 に渡す

### コメント
```bash
# one-line comment
<<COMMENT
multi-line comment
COMMENT
```

## 変数宣言
```bash
# 宣言
v=1            # スペースがあると error
array=(0 1 2)  # スペース区切り
read INPUT     # user 入力

# 展開
echo "${v}"
echo "${array[@]}"

v=`ls`              # コマンドの結果を格納
v=$(( ${v} + 1 ))   # 演算結果を格納
```

## 条件分岐

### 条件式
| 演算子 | 意味 |
| :---: | :--- |
| ! | 否定 |
| x == y | 数値による等式 (-eq でも可） |
| x = y | 文字列による等式 |
| -z x | 文字列 x の長さが 0 |
| -f x | x がファイルである |
| -d x | x がディレクトリである | 

### if
```bash
if [ ! -f ${file} ] ; then        # `[` の両脇はスペース
    式1.
elif [ "${str}" = "hoge" ] ; then   # 変数は "" で囲み NullError を防ぐ
    式2.
else
    式3.
fi
```

### case
```bash
arg="apple"
case ${args} in
    apple)
        echo "apple"
        ;;
    orange)
        echo "orange"
        ;;
    *)
        echo "!(apple or orange)"
        ;;
esac
```


## for / while
```bash
for i in 0 1 2 3
# for (( i=0; i<10; i++ ))
# for i in ${array[@]}
# for file in `ls`
# while [ ${i} < 3 ]
do
    echo ${i}
done
```

```bash
# ファイルを一行ずつ読み込む
while IFS= read -r line
do
    c1
    c2
done < ${FILE}
```

## 文字列操作
```bash
# 文字列処理
str="abacd"

${str/a/k}  # 置換: kbcde
${str//a/k} # 置換: kbkcd

# パス操作
PATH="src/work/hoge.txt"

${PATH##*.} # txt
${PATH##*/} # hoge.txt
${PATH%/*}  # src/work

${PATH#*/}  # work/hoge.txt 最短一致

<<COMMENT
`#` : 最初から検索
`%` : 最後から検索 

`#` : 最短一致
`##`: 最長一致
COMMENT
```

## 関数
```sh
# 定義
hello() {
    # 引数: $1 $2 ...
    echo "Hello $1 !!";    # `;` をつける
}

# 呼び出し
hello "World"   # 引数は後続させる
```

## UNIX commands
- http://www.ritsumei.ac.jp/~tomori/unix.html
- 大変なので都度追加

<details><summary>UNIX commands</summary>

### パス操作
* pwd
* ls
* cp [-r]
* mv
* mkdir -p
* rm -f [-r]

### ファイル・テキスト操作
* file
* wc
* cat
* head -1
* tail -1
* grep -oP
* echo -en
* cut -d "," -f 1
* sort
* paste
* merge
* awk
* sed

### 圧縮・解凍
* tar -xvf [f.tar]	          # .tarの解凍
* gunzip [f.gz]	              # .gzの解凍
* tar -xvzf [f.tar.gz]        #.tar.gzの解凍
* tar -cvzf [f.tar.gz) [path] # .tar.gzの作成

</details>

## リダイレクト
- https://qiita.com/ritukiii/items/b3d91e97b71ecd41d4ea

|redirect|意味|sample|
|:--:|:---|:---|
|<|ファイル内容をコマンドに渡す|
|>|実行結果を新規作成ファイルに書き込み|echo "hoge" > hoge.txt|
|2>|標準エラー出力を新規作成ファイルに書き込み|echo "hoge" 2> hoge.txt|
|&>|標準出力/エラー出力を同一新規作成ファイルに書き込み|echo "hoge" &> hoge.txt|
|2>&1|同上|echo "hoge" > hoge.txt 2>&1|
|>>|実行結果を上書きファイルに書き込み|echo "fuga" >> hoge.txt|
|1>&2||標準出力を標準エラー出力にマージ|
|2>&1||標準エラー出力を標準出力にマージ|


```sh
# ファイル内容をコマンドの標準入力に渡す
command < file

# ファイルにリダイレクト
echo "hoge" > hoge.txt      # mode = w
echo "hoge" >> hoge.txt     # mode = a

>   

```


# OTHER SNIPETS
## pip パッケージサイズ
```bash
pip list | awk '{print $1}' | tail -n +4 | xargs pip show | grep -E 'Location:|Name:' | cut -d ' ' -f 2 | paste -d ' ' - - | awk '{print $2 "/" tolower($1)}' | xargs du -sh 2> /dev/null
```

作成: Miyawaki
最終更新日: 2020.06.13
