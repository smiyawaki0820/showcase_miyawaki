

DCGANの論文にはGANの訓練が安定する工夫として以下があげられている。
- DiscriminatorでPoolingの代わりにStrided Convolutionを使う
- GeneratorはFractional-strided Convolutionを使う
- Generator、DiscriminatorともにBatchNormを使う
- 層が深いときはFC層を除去してGlobal Average Poolingを使う
- GeneratorにはReLUを使う
- ただし出力層のみTanhを使う（今回は画像を0-1標準化したのでSigmoid使用）


reference
- http://aidiary.hatenablog.com/entry/20180304/1520172429
