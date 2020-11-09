# Google Cloud Platform (GCP)
[GPU アクセラレータを使用してインスタンスを実行する](https://cloud.google.com/container-optimized-os/docs/how-to/run-gpus)

## Google Compute Engine (GCE)

### 始める前に
* [Compute Engine での GPU の料金](https://cloud.google.com/compute/gpus-pricing#gpus)
* [GPU の機能・GPU ハードウェアのタイプの詳細](https://cloud.google.com/compute/docs/gpus)
* [Deep Learning VM のドキュメント](https://cloud.google.com/deep-learning-vm/docs/images)
* [リソースの割り当て](https://cloud.google.com/compute/quotas#requesting_additional_quota)


#### インスタンス作成ツール
* Google Cloud Console
* [Compute Engine API](https://cloud.google.com/compute/docs/reference/latest/instances)

### インスタンスの立ち上げ
* https://www.topgate.co.jp/gcp03-google-compute-engine-launch-instance
* [インスタンスへの接続](https://cloud.google.com/compute/docs/instances/connecting-to-instance)
* [GPU ドライバのインストール](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu)
* [GPU の追加・削除](https://cloud.google.com/compute/docs/gpus/add-gpus#create-new-gpu-instance)

### Efficient Open-Domain Question Answering
* [Submission instructions](https://efficientqa.github.io/submit.html#uploading-submissions-and-submitting-to-test)
```bash
# Submit w/ Cloud SDK
$ cd "${SUBMISSION_DIR}"
$ MODEL_TAG=latest
$ gcloud auth login
$ gcloud config set project <your_project_id>
$ gcloud services enable cloudbuild.googleapis.com
$ gcloud builds submit --tag gcr.io/<your_project_id>/${MODEL}:${MODEL_TAG} .
```



# References
* [gcloud reference](https://cloud.google.com/sdk/gcloud/reference?hl=ja)
* [gcloud コマンドライン ツールのクイック リファレンス](https://cloud.google.com/sdk/docs/cheatsheet?hl=ja)
* [gcloud compute](https://cloud.google.com/compute/docs/gcloud-compute)
* [VM と MIG へのコンテナのデプロイ](https://cloud.google.com/compute/docs/containers/deploying-containers?_ga=2.31970941.-299684051.1596777283)
* [GCPでDeep LearningのためのGPU環境を構築する (Zeals.TechBlog)](https://tech.zeals.co.jp/entry/2019/01/08/094054)
* [ディープラーニングイメージで構築する快適・高速な機械学習環境 (slideshare)](https://www.slideshare.net/ooyabuy/ss-114212790)
