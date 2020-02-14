# paper-reading

ACL 採択論文サマリ (xpaperchallenge)
http://xpaperchallenge.org/nlp/summaries/


# template

<a href="http://lafrenze.hatenablog.com/entry/2015/08/04/120205">高速で論文がバリバリ読める落合先生のフォーマットがいい感じだったのでメモ</a>

```template
## 0. 論文
## 1. どんなもの？
## 2. 先行研究と比べてどこがすごい？
## 3. 技術や手法のキモはどこ？
## 4. どうやって有効だと検証した？
## 5. 議論はある？
## 6. 次に読むべき論文は？
```


# 知見
（文章中で一度述語の項となった名詞句は再び項になりやすいという知見）
- センタリング理論[7]
- 統計的観点[8]

# 関連研究

|paper|year|述語間関係|格構造の類似度|名詞句情報（項共有）|顕現性|
|---|---|---|---|---|---|---|
|<a href="https://www.aclweb.org/anthology/P09-2022/">Imamura et al</a>|2009|||||
|<a href="https://www.aclweb.org/anthology/P06-1079/">Iida et al</a>|2006|||||
|<a href="https://www.aclweb.org/anthology/W03-2604/">Iida et al</a>|2003||||O|
|<a href="https://www.anlp.jp/proceedings/annual_meeting/2010/pdf_dir/D3-5.pdf">飯田 徳永</a>|2010||||||



<a href="https://ci.nii.ac.jp/naid/110008583962/">林部ら, 2011</a> 
- 類似した項分布を持つ述語対を手がかりに文脈情報を捉えるため，格構造を「助詞＋述語」と定義して項分布を比較
- 格フレーム（多分）を用いて，その頻度情報から類似度を算出する
- 🤔述語の対象をどうしに限定していたため，`「影響を与える」`といった名詞側が意味をなす句に対して正しい項分布を捉えることができない


<a href="http://www.cl.ecei.tohoku.ac.jp/publications/2013/ono_bthesis.pdf">小野雅之. 2014（卒業論文）</a>
- 格構造の類似度を用いる手法をベースに，意味が捉えられる範囲まで拡張した述語の項分布を扱う
- **機能動詞結合**（`サ変名詞＋格助詞＋機能動詞（実質的な意味を名詞に預けて，自らは専ら文法的な機能を果たす動詞）`：影響を与える，感銘を受ける）を一つの述語とみなす（判定には機能動詞辞書を作成した）
に対して，「サ変名詞の動詞化：[13][14]」ただし態の変換が困難
- 語義曖昧性を持つ述語の格構造の項分布が一様分布になってしまうため，ガ格以外の格を梅田状態で格構造を付与することで曖昧性解消
- inter-ZAR

<a href="https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_action_common_download&item_id=10955&item_no=1&attribute_id=1&file_no=1">飯田ら, </a>
- 先行詞候補が述語の項として使用された回数を素性として用いる

<a href="https://www.aclweb.org/anthology/P09-2022/">Imamura et al., 2009</a>
- 先行詞が項として使用されたことがあるかどうかというbool値を素性として用いる

<a href="https://www.anlp.jp/proceedings/annual_meeting/2010/pdf_dir/D3-5.pdf">飯田 徳永, 2010</a>
- 事態遷移と項共有を使用し，述語対がどの程度項を共有しやすいかのスコアを算出
- PAS の関係タグが付与されたコーパスを利用し，係り受け関係にある述語対に対して，ガ格に共通の項を持つか否かを分類するモデルを作成

<a href="https://www.aclweb.org/anthology/W03-2604/">飯田ら, 2003</a>
- 談話における話題の移り変わりを説明するセンタリング理論をもとに Salience Reference List （項候補をスロットに保持）を用いる
- 文章の先頭から先行し候補が格スロットに該当するか判別（既に格納済の場合，上書き）．述語直前まで繰り返す．
- 主題（ha） > 主語（ga） > 関節目的格（ni） > 直接目的格（o） > その他
