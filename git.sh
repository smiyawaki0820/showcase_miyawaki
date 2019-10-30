#!/usr/bin/env bash

read s Y m d H M S ms ns <<< "$(date + '%s %Y %m %d %H %M %S %3N %9N')"
COMMENT="${m}.${d} $H:$M"

#git pull
git add .
git commit -m ${COMMENT}
git push
