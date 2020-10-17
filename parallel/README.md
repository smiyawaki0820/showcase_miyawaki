## pip パッケージサイズ
```bash
pip list | awk '{print $1}' | tail -n +4 | xargs pip show | grep -E 'Location:|Name:' | cut -d ' ' -f 2 | paste -d ' ' - - | awk '{print $2 "/" tolower($1)}' | xargs du -sh 2> /dev/null
```
