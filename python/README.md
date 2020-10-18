**method chain**
```python
def function(self, *args):
  return self
```

**cudfを使用してread_csv**
```python
import pandas as pd
import cudf
# df = pd.read_csv(fi_csv)
df = cudf.read_csv(fi_csv)
```

**logging snipet**
```python
import logging

logging.basicConfig(
    format='%(asctime)s #%(lineno)s %(levelname)s %(name)s :::  %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
```

* [pathlibチートシート](https://qiita.com/meznat/items/a1cc61edb1e340d0b1a2)
* [ハイパラ管理のすすめ -ハイパーパラメータをHydra+MLflowで管理しよう-](https://ymym3412.hatenablog.com/entry/2020/02/09/034644)
* [chariot](https://github.com/chakki-works/chariot)
