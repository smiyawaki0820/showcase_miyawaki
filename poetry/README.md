# Poetry

## INSTALL
```bash
$ curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
$ export PATH="$HOME/.poetry/bin:$PATH"

# uninstall
$ python get-poetry.py --uninstall
```

## COMMANDs
* [Commands](https://python-poetry.org/docs/cli/)

```bash
$ poetry new
$ poetry init
$ poetry add/remove [package]
$ poetry install --no-dev
$ poetry show
$ poetry shell
$ poetry run [commands]

$ poetry export -f requirements.txt -o requirements.txt
$ pip cache purge  # rm -rf /root/.cache/pip
```

### add options
* `--dev (-D)`: 開発用の依存パッケージとして追加
* `--git`: Git リポジトリの URL を指定
* `--path`: 依存パッケージへのパスを追加
* `--extras (-E)`: 依存パッケージを有効にするための追加パッケージ
* `--optional`: オプションの依存パッケージとして追加
* `--dry-run`: 実行はせず、どんな処理がされるかだけ表示

## pyproject.toml
* [The pyproject.toml file](https://python-poetry.org/docs/pyproject/)



## REFERENCEs
* [Poetry公式ドキュメント](https://poetry.eustace.io/docs/)
* [Poetry: Python の依存関係管理とパッケージングを支援するツール](https://org-technology.com/posts/python-poetry.html)
* [LIFE WITH PYTHON](https://www.lifewithpython.com/2018/12/poetry.html)
* [Poetryを使ったPythonパッケージ開発からPyPI公開まで](https://kk6.hateblo.jp/entry/2018/12/20/124151)
