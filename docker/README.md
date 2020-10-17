# Docker
* CentOS: 4GB
* Ubuntu: 700MB
* Alpine: 100MB


## SNIPETS
```bash
$ docker ps
$ docker info
$ docker images
$ docker rmi ...
$ docker system ...
$ docker system prune
$ docker volume ...
$ docker pull ...
```

## .dockerignore
* 基本的に `.gitignore` と同じ


## Dockerfile
* [docker build](https://docs.docker.jp/engine/reference/commandline/build.html)
* [docker run](https://docs.docker.jp/engine/reference/commandline/run.html)
* [Dockerfile リファレンス](https://docs.docker.jp/engine/reference/builder.html)

```bash
$ vim Dockerfile
$ docker build . -t <tag> -f <file>
$ docker run -it -v <volume> <tag>
```

## dive
* [dive](https://github.com/wagoodman/dive)

```bash
### Ubuntu/Debian ###
$ wget https://github.com/wagoodman/dive/releases/download/v0.9.2/dive_0.9.2_linux_amd64.deb
$ sudo apt install ./dive_0.9.2_linux_amd64.deb

### Mac ###
$ brew install dive

### docker ###
$ docker pull wagoodman/dive
$ alias dive='docker run --rm -it -v /var/run/docker.sock:/var/run/docker.sock -e DOCKER_API_VERSION=1.37 wagoodman/dive:latest'
```


## REFERENCES
* [Docker Compose](https://docs.docker.jp/compose/toc.html)
* [入門 docker. Dockerfileのベストプラクティス](https://y-ohgi.com/introduction-docker/3_production/dockerfile/)
* [Dockerfile を改善するための Best Practice 2019 (slideshare)](https://www.slideshare.net/zembutsu/dockerfile-bestpractices-19-and-advice)
* [Dockerfile を書くためのベストプラクティス解説編 (slideshare)](https://www.slideshare.net/zembutsu/explaining-best-practices-for-writing-dockerfiles)
* [Alpine Linux](https://alpinelinux.org/)
* [pytorch-alpine](https://github.com/petronetto/pytorch-alpine/blob/master/Dockerfile)
