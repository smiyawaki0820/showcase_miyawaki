#!/usr/bin/env bash

read s Y m d H M S ms ns <<< "$(date + '%s %Y %m %d %H ')"

git pull
git add .
git commit -m "${m}.${d} $H:$M"
git push
